#!/bin/bash

DATASET="Tox21_p53"

BATCH_SIZE=64
WGAN_EPOCHS=10000
EPOCHS_DECAY=5000
LR_UPDATE_STEP=1000
PATIENCE=200

G_LR=0.0002
D_LR=0.0002
E_LR=0.0002
ENCODER_EPOCHS=1000
DROPOUT=0.0
GUMBELL_TYPE="no-gumbell"
N_CRITIC=5

if [[ "$DATASET" == "NCI1" || "$DATASET" == "Tox21_MMP" || "$DATASET" == "Tox21_HSE" || "$DATASET" == "Tox21_p53" || "$DATASET" == "IMDB-MULTI" ]]; then
    FEAT="deg"
else
    FEAT="default"
fi

if [ -d "$DATASET" ]; then
    echo "Directory $DATASET exists. Deleting..."
    rm -rf "$DATASET"
fi

echo "Creating directory $DATASET..."
mkdir "$DATASET"

cd "$DATASET"

python3 ../../main.py \
        --DS "$DATASET" \
        --batch_size "$BATCH_SIZE" \
        --feat "$FEAT" \
        --wgan_epochs "$WGAN_EPOCHS" \
        --epochs_decay "$EPOCHS_DECAY" \
        --lr_update_step "$LR_UPDATE_STEP" \
        --patience "$PATIENCE" \
        --g_lr "$G_LR" \
        --d_lr "$D_LR" \
        --e_lr "$E_LR" \
        --encoder_epochs "$ENCODER_EPOCHS" \
        --dropout "$DROPOUT" \
        --gumbell_type "$GUMBELL_TYPE" \
        --n_critic "$N_CRITIC" \
        --plot_loss

