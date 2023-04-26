#visualise data using t-sne plot

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch 
import torchvision.transforms as T
from torchvision import transforms, datasets
from torch.dataset import ImageFolder
from PIL import Image
import os
from model import uda_swin_base_patch4_window7_224_TransReID, uda_swin_small_patch4_window7_224_TransReID   

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
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(data)
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(pca_result)
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
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

def get_flir(dataset_root, batch_size, train):
    """Get FLIR datasets loader
    Args:
        dataset_root (str): path to the dataset folder
        batch_size (int): batch size
        train (bool): create loader for training or test set
    Returns:
        obj: dataloader object for FLIR dataset
    """ 
    # dataset and data loader
    if train:
        pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5776, 0.5776, 0.5776),
                                          std=(0.1319, 0.1319, 0.1319))])
        flir_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'sgada_data/flir/train'),
                                             transform=pre_process)
        

        flir_data_loader = torch.utils.data.DataLoader(
            dataset=flir_dataset,
            batch_size=batch_size,
            shuffle=False, num_workers=4)
    else:
        pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5587, 0.5587, 0.558),
                                          std=(0.1394, 0.1394, 0.1394))])
        flir_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'sgada_data/flir/val'),
                                            transform=pre_process)
        flir_data_loader = torch.utils.data.DataLoader(
            dataset=flir_dataset,
            batch_size=batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    return flir_data_loader


def main():
    #set device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '../logs/pretrain_final/swin_small/coco-flir/flir/1/transformer_10.pth'
    data_path = './data/cocoflir_sample'
    txt_path = './data/cocoflir_sample/flir.txt'

    #get data and labels
    data = []
    labels = []
    transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5776], [0.1394])
    ])
    dataset = ImageDataset(data_path, txt_path, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    #load model
    model = uda_swin_small_patch4_window7_224_TransReID()
    model.load_pretrained(model, model_path)
    #remove last layer of model
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(device)

    for img, label in dataloader:
        img = img.to(device)
        #get features from model
        features = model(img)
        print(features.shape)

        # data.append(features.numpy())
        # labels.append(batch[1].numpy())
    # data = np.concatenate(data, axis=0)
    # labels = np.concatenate(labels, axis=0)
    # #plot t-sne plot
    # plot_tsne(data, labels, 't-sne plot', 'tsne.png')

if __name__ == '__main__':
    main()