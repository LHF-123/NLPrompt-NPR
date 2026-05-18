#!/bin/bash

#cd ../..

set -euo pipefail

DATA=${DATA:-/path/to/datasets}
TRAINER=${TRAINER:-NLPrompt}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-16}
CSC=${CSC:-False}
CTP=${CTP:-end}
SEED_LIST=${SEED_LIST:-"1 2 3"}
LOAD_EPOCH=${LOAD_EPOCH:-50}

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <DATASET> <CFG> [extra train.py opts]"
    exit 1
fi

DATASET=$1
CFG=$2
shift 2

MODEL_ROOT=${MODEL_ROOT:-output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}}

for SEED in ${SEED_LIST}
do
    python train.py \
    --root "${DATA}" \
    --seed "${SEED}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
    --output-dir "output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED}" \
    --model-dir "${MODEL_ROOT}/seed${SEED}" \
    --load-epoch "${LOAD_EPOCH}" \
    --eval-only \
    TRAINER.NLPROMPT.N_CTX "${NCTX}" \
    TRAINER.NLPROMPT.CSC "${CSC}" \
    TRAINER.NLPROMPT.CLASS_TOKEN_POSITION "${CTP}" \
    "$@"
done
