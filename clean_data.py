import torch
import os
import torch
import shutil
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
# from .model.backbones.cvt import ConvolutionalVisionTransformer
from model import make_model
# from .model.backbones.cvt import cvt_21_224_TransReID
from config import cfg
import argparse

file_path="/home/mohamed.imam/Downloads/Projects/CDTrans/data/cocoflir/flir/transformer_best_model.pth"

parser = argparse.ArgumentParser(description="ReID Baseline Training")
parser.add_argument(
    "--config_file", default="", help="path to config file", type=str
)

parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)

parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

if args.config_file != "":
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = make_model(cfg, num_class=3, camera_num=None, view_num = None)
model.load_param_finetune(cfg.TEST.WEIGHT)
model = model.to(device)
# model = ConvolutionalVisionTransformer()
# model.load_state_dict(torch.load(file_path))
# model.eval()


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_name

# Assuming that you already have your PyTorch classification model defined and trained

# Assuming that you already have a directory with sample images that you want to evaluate

# data_root = ['/home/amrin.kareem/Downloads/AI_Project/Old/data/cocoflir/flir/person/', '/home/amrin.kareem/Downloads/AI_Project/Old/data/cocoflir/flir/bicycle/', '/home/amrin.kareem/Downloads/AI_Project/Old/data/cocoflir/flir/car/']
data_root = '/home/mohamed.imam/Downloads/Projects/CDTrans/data/cocoflir/flir/'
sub_dir = ['person', 'bicycle','car']

model.eval()

# Define the transformation that you want to apply to the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

for i in range(len(sub_dir)):
    folder_name = 'bad_images_'+sub_dir[i]
    root_dir = data_root + sub_dir[i]
    conf_file_name = './conf_scores_' + sub_dir[i] + '.txt'

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    # Create the custom dataset
    dataset = CustomDataset(root_dir=root_dir, transform=transform)

    # Create the dataloader for the custom dataset
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    output_file = open(conf_file_name, 'a')
    # Put the model in evaluation mode
   
    x=0

    # Disable gradient computation (not needed for inference, reduces memory consumption)
    with torch.no_grad():
        
        # Iterate over the test dataset/batch
        for images, image_names in dataloader:
            images = images.to(device)
            # Pass the images to the model for inference
            outputs = model(images)
            # Compute the softmax probabilities (confidence scores) of the predictions
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # print (probs)
            # Print the confidence scores of the predictions along with the image name
            for j, sample_probs in enumerate(probs):
                image_name = image_names[j]
                x=x+1
                print (x,root_dir)
                if torch.argmax(sample_probs) != i:
                    # Copy the image to the "bad_images" directory if the index of the maximum value of sample_probs tensor is not 1
                    shutil.copy(os.path.join(root_dir, image_name), os.path.join(folder_name, image_name))
                # print(f"Image: {image_name} - Confidence scores: {sample_probs}")
                # print (sample_probs.shape)
                output_file.write(f"Image: {image_name} - Confidence scores: {sample_probs}\n")