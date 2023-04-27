#visualise data using t-sne plot

import numpy as np
from config import cfg
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch 
import torchvision.transforms as T
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from PIL import Image
import os
from model import uda_swin_base_patch4_window7_224_TransReID, uda_swin_small_patch4_window7_224_TransReID, uda_vit_small_patch16_224_TransReID , make_model  
from tqdm import tqdm
import argparse

def plot_tsne(data, labels, title, save_path):
    #data: numpy array of shape (n_samples, n_features)
    #labels: numpy array of shape (n_samples, )
    #title: string
    #save_path: string
    #return: None
    #plot t-sne plot
    #save plot to save_path
    #plt.figure(figsize=(10, 5))
    #plt.title(title)
    #plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.get_cmap("jet", 10))
    #plt.colorbar(ticks=range(10))
    #plt.clim(-0.5, 9.5)
    #plt.savefig(save_path)
    #plt.show()
    #plt.close()
    # pca = PCA(n_components=50)
    # pca_result = pca.fit_transform(data)
    # print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=plt.cm.get_cmap("jet", 6))
    plt.colorbar(ticks=range(6))
    plt.clim(0, 2)
    plt.savefig(save_path)
    plt.show()
    plt.close()
    return

#torch dataset class to get data and labels from a directory of images and txt file
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, txtfile, transform=None):
        #datadir: string, directory of images
        #txtfile: string, path to txt file
        #transform: torchvision.transforms
        #return: None
        #initialise dataset
        self.datadir = datadir
        self.transform = transform
        self.data = []
        self.labels = []
        with open(txtfile, 'r') as f:
            for line in f:
                img, label = line.split()
                self.data.append(img)
                self.labels.append(int(label))
        return
    def __len__(self):
        #return: int, number of samples in dataset
        return len(self.data)
    def __getitem__(self, idx):
        #idx: int
        #return: tuple of (image, label)
        img = Image.open(os.path.join(self.datadir, self.data[idx]))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def get_dataloader(dataset_dir, batch_size, train):
    
    pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=(0.5587, 0.5587, 0.558),
                                        std=(0.1394, 0.1394, 0.1394))])
    
    dataset = datasets.ImageFolder(root=dataset_dir,
                                        transform=pre_process)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return data_loader


def main(cfg):
    #set device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    model_path = '../logs/pretrain/cvt/transformer_best_model.pth'
    data_path_flir = './data/cocoflir_2000/flir'
    data_path_mscoco = './data/cocoflir_2000/mscoco'
    txt_path = './data/cocoflir_small/flir.txt'

    #get data and labels
    dataloader_flir = get_dataloader(data_path_flir, 128, train=False)
    # dataloader_mscoco = get_data(data_path_mscoco, 128, train=False)

    print('Number of data points in mscoco', len(dataloader_flir)*128)
    # print('Number of data points in flir', len(dataloader_mscoco)*128)

    #load model
    model = make_model(cfg, 3, 0, 0)
    model.load_param_finetune(model_path)

    
    feat_memory1 = []
    label_memory1 = []

    

    model.to(device)
    model.eval()
    for n_iter, (img, vid) in enumerate(tqdm(dataloader_flir)):
        with torch.no_grad():
            img = img.to(device)
            feats = model(img, img)
            feat = feats[1]/(torch.norm(feats[1],2,1,True)+1e-8)
            feat_memory1.append(feat)
            label_memory1.append(vid)

    # feat_memory2 = []
    # label_memory2 = []
    # for n_iter, (img, vid) in enumerate(tqdm(dataloader_mscoco)):
    #     with torch.no_grad():
    #         img = img.to(device)
    #         feats = model(img, img)
    #         feat = feats[1]/(torch.norm(feats[1],2,1,True)+1e-8)
    #         feat_memory2.append(feat)
    #         label_memory2.append(vid+3)

    data = np.concatenate(feat_memory1, axis=0)
    label = np.concatenate(label_memory1, axis=0)
   
    #plot t-sne plot
    plot_tsne(data, label, 't-sne plot', 'tsne.png')

if __name__ == '__main__':
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

    main(cfg)