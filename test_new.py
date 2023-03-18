import torch
import argparse
from utils.logger import setup_logger
import os
from config import cfg
from datasets import make_dataloader
from model import make_model
from tqdm import tqdm
# Define command line arguments
parser = argparse.ArgumentParser(description='Test a pre-trained model on a dataset')
parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to the pre-trained model')
parser.add_argument('--num_classes', type=int, required=True, default=3,
                    help='Number of classes in the dataset')
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
args = parser.parse_args()

if args.config_file != "":
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

device = "cuda"

output_dir = cfg.OUTPUT_DIR
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

logger = setup_logger("reid_baseline", output_dir, if_train=False)
logger.info(args)

if args.config_file != "":
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, 'r') as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
logger.info("Running with config:\n{}".format(cfg))

# Define the test dataset and data loader
train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

# Load the pre-trained model's state dictionary
model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
model.load_param_finetune(cfg.TEST.WEIGHT)
model.to(device)
# Evaluate the model on the test set
model.eval()
class_correct = list(0. for i in range(args.num_classes))
class_total = list(0. for i in range(args.num_classes))
with torch.no_grad():
    for inputs, labels, _, _, _, _ in tqdm(val_loader):
        inputs = inputs.to(device)
        # labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

# Print the accuracy for each class
for i in range(args.num_classes):
    if class_total[i] > 0:
        logger.info('Accuracy of class %d: %2d%% (%2d/%2d)' % (
            i, 100 * class_correct[i] / class_total[i],
            class_correct[i], class_total[i]))
    else:
        print('Accuracy of class %d: N/A (no examples in class)' % (i))
