model=$1
if [ ! -n "$1" ]
then 
    echo 'pelease input the model para: {deit_base, deit_small}'
    exit 8
fi
if [ $model == 'deit_base' ]
then
    model_type='uda_vit_base_patch16_224_TransReID'
    gpus="('0,1')"
else
    model='deit_small'
    model_type='uda_vit_small_patch16_224_TransReID'
    gpus="('0')"
fi
for target_dataset in 'Product' 'Real_World' 'Clipart' # 
do
    python train.py --config_file configs/uda.yml MODEL.DEVICE_ID $gpus \
    OUTPUT_DIR '../logs/uda/'$model'/office-home/Art2'$target_dataset \
    MODEL.PRETRAIN_PATH '../logs/pretrain/'$model'/office-home/Art/transformer_10.pth' \
    DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' \
    DATASETS.ROOT_TRAIN_DIR2 './data/OfficeHomeDataset/'$target_dataset'.txt' \
    DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target_dataset'.txt' \
    DATASETS.NAMES "OfficeHome" DATASETS.NAMES2 "OfficeHome" \
    MODEL.Transformer_TYPE $model_type \

done

OUTPUT_DIR '../logs/pretrain/'$model'/DA_Dataset/RGB_Thermal' \
DATASETS.ROOT_TRAIN_DIR '../DATASETDIR/dataset/mscoco/mscoco.txt' \
DATASETS.ROOT_TEST_DIR '../DATASETDIR/dataset/flir/flir.txt'   \
MODEL.Transformer_TYPE $model_type \
MODEL.PRETRAIN_PATH '../DATASETDIR/dataset/pretrainModel/'$pretrain_model \