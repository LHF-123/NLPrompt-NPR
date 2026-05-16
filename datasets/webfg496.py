import os
import pickle

from PIL import Image

try:
    from PIL import UnidentifiedImageError
except ImportError:
    UnidentifiedImageError = OSError

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_BAD_IMAGE_EXAMPLES = 10


class WebFG496Base(DatasetBase):
    dataset_dir = ""

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        train_classes = self.list_class_dirs(self.train_dir, "train")
        val_classes = self.list_class_dirs(self.val_dir, "val")
        self.check_class_consistency(train_classes, val_classes)
        cname2lab = {c: i for i, c in enumerate(train_classes)}

        train = self.read_split(self.train_dir, "train", cname2lab)
        val = self.read_split(self.val_dir, "val", cname2lab)
        test = val

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(
            train, val, test, subsample=subsample
        )

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def list_class_dirs(split_dir, split):
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Expected WebFG-496 {split} directory at {split_dir}"
            )

        classes = [
            d for d in listdir_nohidden(split_dir, sort=True)
            if os.path.isdir(os.path.join(split_dir, d))
        ]
        if not classes:
            raise RuntimeError(f"No class directories found in {split_dir}")

        return classes

    @staticmethod
    def check_class_consistency(train_classes, val_classes):
        train_set = set(train_classes)
        val_set = set(val_classes)
        if train_set == val_set:
            return

        missing_in_val = sorted(train_set - val_set)
        missing_in_train = sorted(val_set - train_set)
        raise RuntimeError(
            "WebFG-496 train/val class directories do not match. "
            f"Missing in val: {missing_in_val}; "
            f"missing in train: {missing_in_train}"
        )

    def read_split(self, split_dir, split, cname2lab):
        items = []
        bad_images = []
        bad_count = 0

        for cname in sorted(cname2lab.keys(), key=lambda c: cname2lab[c]):
            class_dir = os.path.join(split_dir, cname)
            image_names = [
                im for im in listdir_nohidden(class_dir, sort=True)
                if self.is_image_file(im)
            ]
            class_items = []

            for imname in image_names:
                impath = os.path.join(class_dir, imname)
                if not self.is_valid_image(impath):
                    bad_count += 1
                    if len(bad_images) < MAX_BAD_IMAGE_EXAMPLES:
                        bad_images.append(impath)
                    continue

                item = Datum(
                    impath=impath,
                    label=cname2lab[cname],
                    classname=self.folder_to_classname(cname),
                )
                class_items.append(item)

            if not class_items:
                raise RuntimeError(
                    f"No valid images found for class '{cname}' in {split_dir}"
                )

            items.extend(class_items)

        if bad_count > 0:
            print(
                f"[{self.__class__.__name__}] Skipped {bad_count} "
                f"corrupted image(s) in {split}."
            )
            for impath in bad_images:
                print(f"  {impath}")

        if not items:
            raise RuntimeError(f"No valid WebFG-496 images found in {split_dir}")

        return items

    @staticmethod
    def is_image_file(filename):
        return filename.lower().endswith(IMG_EXTENSIONS)

    @staticmethod
    def is_valid_image(impath):
        try:
            with Image.open(impath) as img:
                img.verify()
            return True
        except (UnidentifiedImageError, OSError, ValueError):
            return False

    @staticmethod
    def folder_to_classname(folder):
        return folder.replace("_", " ")


@DATASET_REGISTRY.register()
class WebAircraft(WebFG496Base):
    dataset_dir = "web-aircraft"


@DATASET_REGISTRY.register()
class WebBird(WebFG496Base):
    dataset_dir = "web-bird"


@DATASET_REGISTRY.register()
class WebCar(WebFG496Base):
    dataset_dir = "web-car"
