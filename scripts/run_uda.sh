model=$1
run=$2
if [ ! -n "$1" ]
then 
    echo 'pelease input the model para: {deit_base, deit_small,swin_small,swin_base}'
    exit 8
fi
if [ $model == 'deit_base' ]
then
    model_type='uda_vit_base_patch16_224_TransReID'
    gpus="('0,1')"
elif [ $model == 'swin_small' ]
then
    model='swin_small'
    model_type='uda_swin_small_patch4_window7_224_TransReID'
    gpus="('0,1')"
elif [ $model == 'swin_base' ]
then
    model='swin_base'
    model_type='uda_swin_base_patch4_window7_224_TransReID'
else
    model='deit_small'
    model_type='uda_vit_small_patch16_224_TransReID'
    gpus="('0')"
fi

python train.py --config_file configs/uda.yml MODEL.DEVICE_ID $gpus \
OUTPUT_DIR '../logs/uda/'$model'/coco-flir/'$run \
DATASETS.ROOT_TRAIN_DIR './data/cocoflir/mscoco.txt' \
DATASETS.ROOT_TRAIN_DIR2 './data/cocoflir/flir.txt' \
DATASETS.ROOT_TEST_DIR './data/cocoflir/flir.txt' \
DATASETS.NAMES "cocoflir" DATASETS.NAMES2 "cocoflir" \
MODEL.Transformer_TYPE $model_type \
MODEL.PRETRAIN_PATH '../logs/pretrain/'$model'/coco-flir/flir/transformer_best_model.pth' \