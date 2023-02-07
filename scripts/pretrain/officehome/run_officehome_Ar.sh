model=$1
if [ ! -n "$1" ]
then 
    echo 'pelease input the model para: {deit_base, deit_small}'
    exit 8
fi
if [ $model == 'deit_base' ]
then
    model_type='vit_base_patch16_224_TransReID'
    pretrain_model='deit_base_distilled_patch16_224-df68dfff.pth'
else
    model='deit_small'
    model_type='vit_small_patch16_224_TransReID'
    pretrain_model='deit_small_distilled_patch16_224-649709d9.pth'
fi
python train.py --config_file configs/pretrain.yml MODEL.DEVICE_ID "('0')" DATASETS.NAMES 'DA_Dataset' \
OUTPUT_DIR '../logs/pretrain/'$model'/DA_Dataset/RGB_Thermal' \
DATASETS.ROOT_TRAIN_DIR '../DATASETDIR/dataset/mscoco/mscoco2.txt' \
DATASETS.ROOT_TEST_DIR '../DATASETDIR/dataset/flir/flir.txt'   \
MODEL.Transformer_TYPE $model_type \
MODEL.PRETRAIN_PATH '../DATASETDIR/dataset/pretrainModel/'$pretrain_model \


