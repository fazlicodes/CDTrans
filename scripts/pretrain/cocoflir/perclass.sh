model=$1
if [ ! -n "$1" ]
then 
    echo 'pelease input the model para: {deit_base, deit_small, swin_base, swin_small}'
    exit 8
fi
if [ $model == 'deit_base' ]
then
    model_type='vit_base_patch16_224_TransReID'
    pretrain_model='deit_base_distilled_patch16_224-df68dfff.pth'
elif [ $model == 'swin_small' ]
then
    model='swin_small'
    model_type='swin_small_patch4_window7_224_TransReID'
    pretrain_model='swin_small_patch4_window7_224_22k.pth'
elif [ $model == 'deit_small' ]
then
    model='deit_small'
    model_type='vit_small_patch16_224_TransReID'
    pretrain_model='deit_small_distilled_patch16_224-649709d9.pth'
else
    model='swin_base'
    model_type='swin_base_patch4_window7_224_TransReID'
    pretrain_model='swin_base_patch4_window7_224_22k.pth'
fi
python test_new.py --config_file configs/pretrain.yml --num_classes 3 --model_path '../logs/pretrain/'$model'/coco-flir/mscoco/transformer_best_model.pth' MODEL.DEVICE_ID "('0')" DATASETS.NAMES 'flir' \
OUTPUT_DIR '../logs/target/perclass/'$model'/coco-flir/flir' \
DATASETS.ROOT_TRAIN_DIR './data/cocoflir/train_labels.txt' \
DATASETS.ROOT_TEST_DIR './data/cocoflir/val_labels.txt'   \
MODEL.Transformer_TYPE $model_type \
MODEL.PRETRAIN_PATH './data/pretrainModel/'$pretrain_model \
TEST.WEIGHT '../logs/pretrain/'$model'/coco-flir/mscoco/transformer_best_model.pth' \


