model=$1
if [ ! -n "$1" ]
then 
    echo 'pelease input the model para: {deit_base, deit_small, cvt}'
    exit 8
fi
if [ $model == 'cvt' ]
then
    model_type='cvt_21_224_TransReID'
    pretrain_model='CvT-21-224x224-IN-1k.pth'
else
    model='deit_small'
    model_type='vit_small_patch16_224_TransReID'
    pretrain_model='deit_small_distilled_patch16_224-649709d9.pth'
fi
python train.py --config_file configs/pretrain.yml MODEL.DEVICE_ID "('0')" DATASETS.NAMES 'cocoflir' \
OUTPUT_DIR '../logs/pretrain2/'$model'/coco-flir/mscoco' \
DATASETS.ROOT_TRAIN_DIR '/home/amrin.kareem/Downloads/AI_Project/Old/data/cocoflir/mscoco.txt' \
DATASETS.ROOT_TEST_DIR '/home/amrin.kareem/Downloads/AI_Project/Old/data/cocoflir/flir.txt'   \
MODEL.Transformer_TYPE $model_type \
MODEL.PRETRAIN_PATH './data/pretrainModel/'$pretrain_model \


