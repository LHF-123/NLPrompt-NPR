import datetime
import math
import os.path as osp
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import AverageMeter, MetricMeter, load_checkpoint, load_pretrained_weights
from trainers.nlprompt import TextEncoder, load_clip_to_cpu

_tokenizer = _Tokenizer()


class NPRPromptLearner(nn.Module):
    """Learnable unified context tokens for CLIP prompt tuning."""

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        npr_cfg = cfg.TRAINER.NPR
        n_cls = len(classnames)
        n_ctx = npr_cfg.PROMPT_LENGTH
        ctx_init = npr_cfg.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Initializing a generic NPR context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial NPR context: "{prompt_prefix}"')
        print(f"Number of NPR context tokens: {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        class_names = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in class_names]
        prompts = [prompt_prefix + " " + name + "." for name in class_names]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = npr_cfg.CLASS_TOKEN_POSITION

    def forward(self, ctx_override=None):
        ctx = self.ctx if ctx_override is None else ctx_override.to(self.token_prefix.dtype)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1]
                class_i = suffix[i:i + 1, :name_len]
                suffix_i = suffix[i:i + 1, name_len:]
                ctx_i = ctx[i:i + 1]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i[:, :half_n_ctx],
                        class_i,
                        ctx_i[:, half_n_ctx:],
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1]
                class_i = suffix[i:i + 1, :name_len]
                suffix_i = suffix[i:i + 1, name_len:]
                ctx_i = ctx[i:i + 1]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError(f"Unsupported class token position: {self.class_token_position}")

        return prompts


class NPRPromptModules(nn.Module):
    """Container saved by checkpoints: clean/noise prompts plus clean EMA."""

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.clean = NPRPromptLearner(cfg, classnames, clip_model)
        self.noise = NPRPromptLearner(cfg, classnames, clip_model)
        self.register_buffer("clean_ctx_ema", self.clean.ctx.detach().float().clone())

    @torch.no_grad()
    def update_clean_ema(self, momentum):
        ctx = self.clean.ctx.detach().float()
        if self.clean_ctx_ema.shape != ctx.shape:
            self.clean_ctx_ema = ctx.clone()
            return
        self.clean_ctx_ema.mul_(momentum).add_(ctx, alpha=1.0 - momentum)


class NPRCLIP(nn.Module):
    """CLIP wrapper with clean/noise learnable prompt branches."""

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_modules = NPRPromptModules(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.classnames = [name.replace("_", " ") for name in classnames]

    def encode_image(self, images):
        image_features = self.image_encoder(images.type(self.dtype))
        return image_features / image_features.norm(dim=-1, keepdim=True)

    def encode_prompt(self, prompt="clean"):
        if prompt == "clean":
            learner = self.prompt_modules.clean
            prompts = learner()
        elif prompt == "noise":
            learner = self.prompt_modules.noise
            prompts = learner()
        elif prompt == "ema":
            learner = self.prompt_modules.clean
            prompts = learner(ctx_override=self.prompt_modules.clean_ctx_ema)
        else:
            raise ValueError(f"Unsupported prompt branch: {prompt}")

        tokenized_prompts = learner.tokenized_prompts.to(prompts.device)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_hard_prompts(self, templates):
        features = []
        device = self.logit_scale.device
        for template in templates:
            prompts = [template.format(name) for name in self.classnames]
            tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
            embedding = self.token_embedding(tokenized).type(self.dtype)
            text_features = self.text_encoder(embedding, tokenized)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            features.append(text_features)
        text_features = torch.stack(features, dim=0).mean(dim=0)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def forward(self, images, prompt="clean", return_features=False):
        image_features = self.encode_image(images)
        text_features = self.encode_prompt(prompt)
        logits = self.logit_scale.exp() * image_features @ text_features.t()
        if return_features:
            return logits, image_features
        return logits


@TRAINER_REGISTRY.register()
class CLIPZeroShot(TrainerX):
    """Zero-shot CLIP evaluator using the NPR hard prompt templates."""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.NPR.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP zero-shot model (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.NPR.PREC in ["fp32", "amp"] or self.device.type == "cpu":
            clip_model.float()
        self.model = NPRCLIP(cfg, classnames, clip_model)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            self.hard_text_features = self.model.encode_hard_prompts(NPR.HARD_PROMPT_TEMPLATES)

    def load_model(self, directory, epoch=None):
        print("CLIPZeroShot does not load a checkpoint")

    def model_inference(self, input):
        image_features = self.model.encode_image(input)
        return self.model.logit_scale.exp() * image_features @ self.hard_text_features.t()

    def forward_backward(self, batch):
        raise RuntimeError("CLIPZeroShot only supports --eval-only")


@TRAINER_REGISTRY.register()
class NPR(TrainerX):
    """Semantic-neighborhood noise-aware prompt learning."""

    HARD_PROMPT_TEMPLATES = (
        "a photo of a {}.",
        "a fine-grained photo of a {}.",
        "a close-up photo of a {}.",
    )

    def __init__(self, cfg):
        self.npr_cfg = cfg.TRAINER.NPR
        self.prec = self.npr_cfg.PREC
        super().__init__(cfg)
        self.scaler = GradScaler() if self.prec == "amp" else None
        self.q_ema = None
        self.clean_mask = None
        self.id_mask = None
        self.ood_mask = None

    def check_cfg(self, cfg):
        assert cfg.TRAINER.NPR.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.NPR.STATIC_MODE in ["full_softmax", "neighbor"]
        assert cfg.TRAINER.NPR.SPLIT_MODE == "classwise_quantile"
        assert cfg.TRAINER.NPR.EVAL_PROMPT == "clean"
        assert cfg.TRAINER.NPR.NOISE_TARGET in ["entropy", "fit_hard", "clean_reject"]

    def build_model(self):
        cfg = self.cfg
        prec = cfg.TRAINER.NPR.PREC
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if prec in ["fp32", "amp"] or self.device.type == "cpu":
            clip_model.float()

        print("Building NPR CLIP")
        self.model = NPRCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in CLIP image/text encoders")
        for name, param in self.model.named_parameters():
            if not name.startswith("prompt_modules."):
                param.requires_grad_(False)

        if not cfg.TRAINER.NPR.USE_NOISE_PROMPT:
            for param in self.model.prompt_modules.noise.parameters():
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_modules.clean, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(
            [p for p in self.model.prompt_modules.parameters() if p.requires_grad],
            cfg.OPTIM,
        )
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_modules", self.model.prompt_modules, self.optim, self.sched)

        with torch.no_grad():
            self.hard_text_features = self.model.encode_hard_prompts(self.HARD_PROMPT_TEMPLATES)
            self.semantic_neighbors = self._build_semantic_neighbors(self.hard_text_features)

    @staticmethod
    def _safe_log(x):
        return torch.log(x.clamp_min(1e-12))

    def _build_semantic_neighbors(self, text_features):
        k = min(self.npr_cfg.NEIGHBOR_K, text_features.size(0) - 1)
        sim = text_features @ text_features.t()
        sim.fill_diagonal_(-float("inf"))
        return sim.topk(k=k, dim=1).indices

    def _compute_q_sem(self, hard_logits_tau, p_hard, labels):
        if self.npr_cfg.STATIC_MODE == "full_softmax":
            return p_hard[torch.arange(labels.size(0), device=labels.device), labels]

        neighbors = self.semantic_neighbors[labels]
        candidates = torch.cat([labels.unsqueeze(1), neighbors], dim=1)
        candidate_logits = hard_logits_tau.gather(1, candidates)
        return F.softmax(candidate_logits, dim=1)[:, 0]

    def _combine_reliability(self, q_sem, p_hard, p_ema, labels):
        q = q_sem if self.npr_cfg.USE_Q_SEM else None

        if self.npr_cfg.USE_Q_DYN:
            p_y = p_ema[torch.arange(labels.size(0), device=labels.device), labels]
            entropy = -(p_ema * self._safe_log(p_ema)).sum(dim=1)
            q_dyn = p_y * torch.exp(-entropy)
            q = q_dyn if q is None else torch.sqrt((q * q_dyn).clamp_min(1e-12))

        if self.npr_cfg.USE_Q_AGREE:
            q_agree = (p_hard * p_ema).sum(dim=1)
            q = q_agree if q is None else q * q_agree

        if q is None:
            q = q_sem

        return q.clamp(0.0, 1.0)

    @torch.no_grad()
    def before_epoch(self):
        self.set_model_mode("eval")
        num_samples = len(self.train_loader_x.dataset)
        all_q = torch.zeros(num_samples, device=self.device)
        all_q_in = torch.zeros(num_samples, device=self.device)
        all_labels = torch.zeros(num_samples, dtype=torch.long, device=self.device)

        hard_text = self.hard_text_features
        ema_text = self.model.encode_prompt("ema")

        for batch in self.train_loader_x:
            images = batch["img"].to(self.device)
            labels = batch["label"].to(self.device)
            index = batch["index"].to(self.device)

            with autocast(enabled=self.device.type == "cuda" and self.prec in ["fp16", "amp"]):
                image_features = self.model.encode_image(images)
                hard_logits_tau = image_features @ hard_text.t() / self.npr_cfg.TAU
                p_hard = F.softmax(hard_logits_tau, dim=1)
                ema_logits_tau = image_features @ ema_text.t() / self.npr_cfg.TAU
                p_ema = F.softmax(ema_logits_tau, dim=1)

            q_sem = self._compute_q_sem(hard_logits_tau, p_hard, labels)
            q = self._combine_reliability(q_sem, p_hard, p_ema, labels)
            q_in = 0.5 * (p_hard.max(dim=1).values + p_ema.max(dim=1).values)

            all_q[index] = q.float()
            all_q_in[index] = q_in.float()
            all_labels[index] = labels

        if self.q_ema is None or self.q_ema.numel() != num_samples:
            self.q_ema = all_q.clone()
        else:
            rho = self.npr_cfg.Q_EMA_MOMENTUM
            self.q_ema.mul_(rho).add_(all_q, alpha=1.0 - rho)

        self._build_split_masks(self.q_ema, all_q_in, all_labels)
        print(
            "NPR reliability "
            f"mean={self.q_ema.mean().item():.4f} "
            f"min={self.q_ema.min().item():.4f} "
            f"max={self.q_ema.max().item():.4f}"
        )

    def _build_split_masks(self, q_values, q_in, labels):
        num_samples = q_values.numel()
        clean_mask = torch.zeros(num_samples, dtype=torch.bool, device=self.device)
        id_mask = torch.zeros(num_samples, dtype=torch.bool, device=self.device)
        ood_mask = torch.zeros(num_samples, dtype=torch.bool, device=self.device)

        for cls in labels.unique(sorted=True):
            cls_idx = torch.nonzero(labels == cls, as_tuple=False).flatten()
            cls_q = q_values[cls_idx]
            order = torch.argsort(cls_q, descending=True)
            sorted_idx = cls_idx[order]
            n_cls = sorted_idx.numel()
            n_clean = max(1, int(math.ceil(n_cls * self.npr_cfg.CLEAN_RATIO)))
            clean_idx = sorted_idx[:n_clean]
            clean_mask[clean_idx] = True

            if self.npr_cfg.USE_OOD_BRANCH:
                n_ood = int(math.floor(n_cls * self.npr_cfg.OOD_RATIO))
                if n_ood > 0:
                    low_idx = sorted_idx[-n_ood:]
                    low_idx = low_idx[q_in[low_idx] < self.npr_cfg.TAU_IN]
                    ood_mask[low_idx] = True

            if self.npr_cfg.USE_ID_PLL:
                id_idx = sorted_idx[~clean_mask[sorted_idx] & ~ood_mask[sorted_idx]]
                id_mask[id_idx] = True

        self.clean_mask = clean_mask
        self.id_mask = id_mask
        self.ood_mask = ood_mask
        print(
            "NPR split "
            f"clean={clean_mask.sum().item()} "
            f"id={id_mask.sum().item()} "
            f"ood={ood_mask.sum().item()} "
            f"ignored={(~(clean_mask | id_mask | ood_mask)).sum().item()}"
        )

    def _prompt_distribution(self, image_features, prompt):
        text_features = self.model.encode_prompt(prompt)
        logits = image_features @ text_features.t() / self.npr_cfg.TAU
        return F.softmax(logits, dim=1)

    def _candidate_mask(self, labels, p_hard, p_ema):
        batch_size = labels.size(0)
        num_classes = p_hard.size(1)
        topk = min(self.npr_cfg.CANDIDATE_TOPK, num_classes)
        mask = torch.zeros(batch_size, num_classes, dtype=torch.bool, device=labels.device)
        mask.scatter_(1, labels.unsqueeze(1), True)
        mask.scatter_(1, p_hard.topk(k=topk, dim=1).indices, True)
        mask.scatter_(1, p_ema.topk(k=topk, dim=1).indices, True)
        mask.scatter_(1, self.semantic_neighbors[labels], True)
        return mask

    @staticmethod
    def _partial_label_loss(logits, candidate_mask):
        probs = F.softmax(logits, dim=1)
        mass = (probs * candidate_mask.float()).sum(dim=1).clamp_min(1e-12)
        return -torch.log(mass).mean()

    @staticmethod
    def _high_entropy_loss(logits):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        num_classes = logits.size(1)
        return (probs * (log_probs + math.log(num_classes))).sum(dim=1).mean()

    def _orthogonal_loss(self):
        clean_ctx = self.model.prompt_modules.clean.ctx.float().reshape(-1, self.model.prompt_modules.clean.ctx.size(-1))
        noise_ctx = self.model.prompt_modules.noise.ctx.float().reshape(-1, self.model.prompt_modules.noise.ctx.size(-1))
        clean_ctx = F.normalize(clean_ctx, dim=1)
        noise_ctx = F.normalize(noise_ctx, dim=1)
        return (clean_ctx @ noise_ctx.t()).pow(2).mean()

    def _compute_loss(self, images, labels, index):
        logits_clean, image_features = self.model(images, prompt="clean", return_features=True)
        clean_batch = self.clean_mask[index]
        id_batch = self.id_mask[index]
        ood_batch = self.ood_mask[index]

        zero = logits_clean.new_zeros(())
        loss = zero
        loss_clean = zero
        loss_id = zero
        loss_ood = zero
        loss_ortho = zero
        has_active_loss = False

        if clean_batch.any():
            loss_clean = F.cross_entropy(logits_clean[clean_batch], labels[clean_batch])
            loss = loss + loss_clean
            has_active_loss = True

        if id_batch.any() and self.npr_cfg.USE_ID_PLL:
            with torch.no_grad():
                hard_logits_tau = image_features.detach() @ self.hard_text_features.t() / self.npr_cfg.TAU
                p_hard = F.softmax(hard_logits_tau, dim=1)
                p_ema = self._prompt_distribution(image_features.detach(), "ema")
                candidates = self._candidate_mask(labels, p_hard, p_ema)
            loss_id = self._partial_label_loss(logits_clean[id_batch], candidates[id_batch])
            loss = loss + self.npr_cfg.LAMBDA_ID * loss_id
            has_active_loss = True

        if ood_batch.any() and self.npr_cfg.USE_NOISE_PROMPT:
            if self.npr_cfg.NOISE_TARGET == "fit_hard":
                logits_noise = self.model(images[ood_batch], prompt="noise")
                loss_ood = F.cross_entropy(logits_noise, labels[ood_batch])
            elif self.npr_cfg.NOISE_TARGET == "clean_reject":
                probs_clean = F.softmax(logits_clean[ood_batch], dim=1)
                p_y = probs_clean[torch.arange(probs_clean.size(0), device=labels.device), labels[ood_batch]]
                loss_ood = -torch.log((1.0 - p_y).clamp_min(1e-12)).mean()
            elif self.npr_cfg.USE_OOD_ENTROPY:
                logits_noise = self.model(images[ood_batch], prompt="noise")
                loss_ood = self._high_entropy_loss(logits_noise)
            loss = loss + self.npr_cfg.LAMBDA_OOD * loss_ood
            has_active_loss = True

        if self.npr_cfg.USE_NOISE_PROMPT and self.npr_cfg.USE_ORTHO:
            loss_ortho = self._orthogonal_loss()
            loss = loss + self.npr_cfg.LAMBDA_ORTHO * loss_ortho
            has_active_loss = True

        return loss, logits_clean, {
            "loss": loss.item(),
            "loss_clean": loss_clean.item(),
            "loss_id": loss_id.item(),
            "loss_ood": loss_ood.item(),
            "loss_ortho": loss_ortho.item(),
            "n_clean": float(clean_batch.sum().item()),
            "n_id": float(id_batch.sum().item()),
            "n_ood": float(ood_batch.sum().item()),
            "skip": 0.0 if has_active_loss else 1.0,
        }

    def forward_backward(self, batch):
        images, labels, _, index = self.parse_batch_train(batch)

        if self.prec == "amp":
            with autocast():
                loss, logits, summary = self._compute_loss(images, labels, index)
            if summary["skip"] > 0:
                summary["acc"] = 0.0
                return summary
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss, logits, summary = self._compute_loss(images, labels, index)
            if summary["skip"] > 0:
                summary["acc"] = 0.0
                return summary
            self.model_backward_and_update(loss)

        self.model.prompt_modules.update_clean_ema(self.npr_cfg.PROMPT_EMA_MOMENTUM)
        summary["acc"] = compute_accuracy(logits, labels)[0].item()
        return summary

    def parse_batch_train(self, batch):
        images = batch["img"].to(self.device)
        labels = batch["label"].to(self.device)
        gt_labels = batch["gttarget"].to(self.device)
        index = batch["index"].to(self.device)
        return images, labels, gt_labels, index

    def model_inference(self, input):
        return self.model(input, prompt="clean")

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in self.get_model_names():
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            for key in list(state_dict.keys()):
                if "token_prefix" in key or "token_suffix" in key:
                    del state_dict[key]

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta = str(datetime.timedelta(seconds=int(batch_time.avg * nb_remain)))
                info = [
                    f"epoch [{self.epoch + 1}/{self.max_epoch}]",
                    f"batch [{self.batch_idx + 1}/{self.num_batches}]",
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})",
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})",
                    f"{losses}",
                    f"lr {self.get_current_lr():.4e}",
                    f"eta {eta}",
                ]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)
            end = time.time()
