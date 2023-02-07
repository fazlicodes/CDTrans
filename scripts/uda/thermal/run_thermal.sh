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
    pretrain_model='deit_small_distilled_patch16_224-649709d9.pth'
    gpus="('0')"
fi

for target_dataset in 'flir' # 
do
    python train.py --config_file configs/uda.yml MODEL.DEVICE_ID $gpus \
    OUTPUT_DIR '../logs/uda/'$model'/DA_Dataset/RGB_Thermal'  \
    MODEL.PRETRAIN_PATH '../DATASETDIR/dataset/pretrainModel/'$pretrain_model \
    DATASETS.ROOT_TRAIN_DIR '../DATASETDIR/dataset/mscoco/mscoco.txt' \
    DATASETS.ROOT_TRAIN_DIR2 '../DATASETDIR/dataset/flir/flir.txt'   \
    DATASETS.ROOT_TEST_DIR '../DATASETDIR/dataset/flir/flir.txt'   \
    DATASETS.NAMES "DA_Dataset" DATASETS.NAMES2 "DA_Dataset" \
    MODEL.Transformer_TYPE $model_type \

done



