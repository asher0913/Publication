#!/bin/bash
# =============================================================================
# run_exp.sh — FaceScrub main experiment (Slot Cross-Attention CEM)
#
# This is the exact configuration that produced the headline result
# reported as "My method (Attn-CEM)" in the dissertation main table
# (Acc 79.49%, MSE 0.0274, SSIM 0.518, PSNR 15.63 dB).
#
# Pipeline:
#   Phase 1  training           — main_MIA.py, 300 epochs
#   Phase 2  decoding-based MIA — main_test_MIA.py, 50 attack epochs
#
# Hyper-parameters (see dissertation §Methodology, Table:dev-iterations):
#   bottleneck = noRELU_C16S1   (16-channel smashed data)
#   lambda     = 10             (CEM weight)
#   sigma      = 0.025          (Gaussian-KL noise)
#   AT_reg_str = 0.15           (slot-attention regulariser weight)
#   slots=8, slot_dim=64, iters=3, bank=64, warmup=3, heads=4
#
# Usage:   bash run_exp.sh [GPU_ID]       (default GPU_ID = 0)
# Output:  saves/facescrub/SCA_new_cemfixed_acc80c16v2_vt0.15/...
#          runlog_acc80_<timestamp>/exp01.log
# =============================================================================

set -uo pipefail

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! python -c "import sys; assert sys.version_info[0] >= 3" 2>/dev/null; then
    echo "ERROR: 'python' is not Python 3."
    echo "       Activate your conda environment first, then re-run."
    exit 1
fi
PY=python

RUN_TS="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="${SCRIPT_DIR}/runlog_acc80_${RUN_TS}"
mkdir -p "${RUN_DIR}"
MASTER_LOG="${RUN_DIR}/master.log"

lm()   { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "${MASTER_LOG}"; }
lexp() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" \
             | tee -a "${EXP_LOG}" \
             | tee -a "${MASTER_LOG}"; }

lm "======================================================================"
lm " run_exp.sh  |  FaceScrub main experiment (SCA-CEM, C16)  |  GPU ${GPU_ID}"
lm " Run dir  :  ${RUN_DIR}"
lm "======================================================================"

# ── parameters ───────────────────────────────────────────────────────────────
ARCH=vgg11_bn_sgm
BATCH_SIZE=256
NUM_CLIENT=1
RANDOM_SEED=125
CUTLAYER=4
BOTTLENECK=noRELU_C16S1
DATASET=facescrub
SCHEME=V2_epoch
REGULARIZATION=Gaussian_kl
LOG_ENTROPY=1
AT_REG=SCA_new
AT_REG_STR=0.15
TRAIN_AE=res_normN4C64
TEST_AE=res_normN8C64
GAN_LOSS=SSIM
SSIM_THR=0.5
LOCAL_LR=-1
ATTACK_EPOCHS=50
LR=0.05
EPOCHS=300

# ── weakened defense (v2) ──────────────────────────────────────────────────
LAMBD=10                      # v1=14 → v2=10
NOISE=0.025                   # v1=0.035 → v2=0.025
VT=0.15
LS=1.0
SLOTS=8
ITERS=3
BANK=64
SDIM=64
WARMUP=3
HEADS=4

EID="exp01"
EXP_LOG="${RUN_DIR}/${EID}.log"

EFF=$(awk "BEGIN{ printf \"%.1f\", ${LAMBD} * ${LS} }")
THR=$(awk "BEGIN{ printf \"%.2e\", ${VT} * ${NOISE}^2 }")

FOLDER="saves/facescrub/${AT_REG}_cemfixed_acc80c16v2_vt${VT}"
FNAME="l${LAMBD}_n${NOISE}_ep${EPOCHS}_vt${VT}_ls${LS}_sl${SLOTS}_it${ITERS}_bk${BANK}_sd${SDIM}_wu${WARMUP}_at${AT_REG_STR}_bn${BOTTLENECK}"

{
    printf '\n'
    printf '%.0s=' {1..72}; printf '\n'
    printf ' FaceScrub main experiment (SCA-CEM, C16 bottleneck)\n'
    printf '   lambd=%-4s  noise=%-5s  var_thr=%-5s  loss_scale=%s  at_reg_str=%s\n' \
           "${LAMBD}" "${NOISE}" "${VT}" "${LS}" "${AT_REG_STR}"
    printf '   bottleneck=%s\n' "${BOTTLENECK}"
    printf '   effective_strength=%-5s  threshold=%s\n' "${EFF}" "${THR}"
    printf '   slots=%-3s  slot_dim=%s  iters=%s  bank=%s  warmup=%s  epochs=%s\n' \
           "${SLOTS}" "${SDIM}" "${ITERS}" "${BANK}" "${WARMUP}" "${EPOCHS}"
    printf '   Expected: Acc 79.49%%, MSE 0.0274, SSIM 0.518, PSNR 15.63 dB\n'
    printf '%.0s=' {1..72}; printf '\n'
} | tee -a "${EXP_LOG}" | tee -a "${MASTER_LOG}"

# ── Phase 1: Training ────────────────────────────────────────────────────────
lexp "[${EID}] >>>>>> TRAINING START  $(date)"

${PY} main_MIA.py \
    --arch="${ARCH}" \
    --cutlayer="${CUTLAYER}" \
    --batch_size="${BATCH_SIZE}" \
    --filename="${FNAME}" \
    --num_client="${NUM_CLIENT}" \
    --num_epochs="${EPOCHS}" \
    --dataset="${DATASET}" \
    --scheme="${SCHEME}" \
    --regularization="${REGULARIZATION}" \
    --regularization_strength="${NOISE}" \
    --log_entropy="${LOG_ENTROPY}" \
    --AT_regularization="${AT_REG}" \
    --AT_regularization_strength="${AT_REG_STR}" \
    --random_seed="${RANDOM_SEED}" \
    --learning_rate="${LR}" \
    --lambd="${LAMBD}" \
    --gan_AE_type="${TRAIN_AE}" \
    --gan_loss_type="${GAN_LOSS}" \
    --local_lr="${LOCAL_LR}" \
    --bottleneck_option="${BOTTLENECK}" \
    --folder="${FOLDER}" \
    --ssim_threshold="${SSIM_THR}" \
    --var_threshold="${VT}" \
    --attention_num_slots="${SLOTS}" \
    --attention_num_heads="${HEADS}" \
    --attention_num_iterations="${ITERS}" \
    --attention_loss_scale="${LS}" \
    --attention_warmup_epochs="${WARMUP}" \
    --attention_bank_size="${BANK}" \
    --attention_slot_dim="${SDIM}" \
    2>&1 | tee -a "${EXP_LOG}" | tee -a "${MASTER_LOG}"
TRAIN_RC=${PIPESTATUS[0]}

if [ "${TRAIN_RC}" -ne 0 ]; then
    lexp "[${EID}] !!! TRAINING FAILED (exit=${TRAIN_RC})  $(date)"
    lm "Aborting — training failed."
    exit 1
fi

lexp "[${EID}] <<<<<< TRAINING OK  $(date)"

# ── Phase 2: MIA Attack ──────────────────────────────────────────────────────
lexp "[${EID}] >>>>>> ATTACK START  $(date)"

${PY} main_test_MIA.py \
    --arch="${ARCH}" \
    --cutlayer="${CUTLAYER}" \
    --batch_size="${BATCH_SIZE}" \
    --filename="${FNAME}" \
    --num_client="${NUM_CLIENT}" \
    --num_epochs="${EPOCHS}" \
    --dataset="${DATASET}" \
    --scheme="${SCHEME}" \
    --regularization="${REGULARIZATION}" \
    --regularization_strength="${NOISE}" \
    --log_entropy="${LOG_ENTROPY}" \
    --AT_regularization="${AT_REG}" \
    --AT_regularization_strength="${AT_REG_STR}" \
    --random_seed="${RANDOM_SEED}" \
    --gan_loss_type="${GAN_LOSS}" \
    --attack_epochs="${ATTACK_EPOCHS}" \
    --bottleneck_option="${BOTTLENECK}" \
    --folder="${FOLDER}" \
    --var_threshold="${VT}" \
    --attention_num_slots="${SLOTS}" \
    --attention_num_heads="${HEADS}" \
    --attention_num_iterations="${ITERS}" \
    --attention_loss_scale="${LS}" \
    --attention_warmup_epochs="${WARMUP}" \
    --attention_bank_size="${BANK}" \
    --attention_slot_dim="${SDIM}" \
    --average_time=1 \
    --gan_AE_type="${TEST_AE}" \
    --test_best \
    2>&1 | tee -a "${EXP_LOG}" | tee -a "${MASTER_LOG}"
ATTACK_RC=${PIPESTATUS[0]}

if [ "${ATTACK_RC}" -ne 0 ]; then
    lexp "[${EID}] !!! ATTACK FAILED (exit=${ATTACK_RC})  $(date)"
    exit 1
fi

lexp "[${EID}] <<<<<< ATTACK OK  $(date)"

lm ""
lm "======================================================================"
lm " DONE — experiment complete"
lm "   Log: ${EXP_LOG}"
lm "   Expected: Acc 79.49%, MSE 0.0274, SSIM 0.518, PSNR 15.63 dB"
lm "======================================================================"
