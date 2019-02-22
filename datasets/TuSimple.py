"""Dataset setting and data loader for MNIST_M.

Modified from
https://github.com/mingyuliutw/CoGAN_PyTorch/blob/master/src/dataset_usps.py
"""

import gzip
import os
import pickle
import urllib
import numpy as np
from scipy.io import loadmat

import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import Image
from misc import params


class OFFICE31(data.Dataset):
    """OFFICE Dataset.

    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    #url = "https://ufldl.stanford.edu/housenumbers/"

    sets = ['amazon', 'caltech10', 'dslr', 'webcam', 'caltech']

    def __init__(self, root, sets="caltech", train=True, transform=None, download=False):
        """Init OFFICE dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "Hnet/pytorch-arda"
        self.sets = sets
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()
        print len(self.train_labels)
        if self.train:
            total_num_samples = self.train_data.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels
        #self.train_data *= 255.0
        #print(self.train_data.shape)
        self.train_data = self.train_data.transpose(
            (0, 3, 2, 1))  # convert to HWC
        #print('train_data shape', self.train_data.shape)
     

    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]

        if self.transform is not None:
            img = self.transform(img)
        label = torch.FloatTensor(label)
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        print os.path.join(self.root, self.filename)
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        print(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        images = []
        labels = []
        
        train_label = open(os.path.join(filename, 'data1/TuSimple_center_angle_2.txt'), 'r')
        transform = transforms.Compose([transforms.Resize((320,176))
                                      #transforms.RandomResizedCrop(224),
                                      #transforms.RandomHorizontalFlip()
                                      ])
        if self.train:
            for f in train_label:
                train_file = filename + '/data1' + f.split('  ')[0]
                uu = f.replace('\n', '').split('  ')[2::]
                
                points1 = []   
                points2 = []
                points3 = []
                points4 = []
                #points = []
                for i in range(len(uu)):
                    data = uu[i].split(' ')
                    #print data
                    point_x = float(data[5])
                    point_y = float(data[6])
                    position = int(data[7])
                    if position == 2:
                        points2.append([point_x, point_y]) 
                
                    if position == 3:
                        points3.append([point_x, point_y]) 
                
                    if position == 1:
                        points1.append([point_x, point_y]) 
                
                    if position == 4:
                        points4.append([point_x, point_y]) 
                points1 = np.array(points1, dtype = np.float32)
                points2 = np.array(points2, dtype = np.float32)
                points3 = np.array(points3, dtype = np.float32)
                points4 = np.array(points4, dtype = np.float32)
                points = np.zeros([4,300, 2], dtype = np.float32)
                #print 'before: ', points1
                if points1.shape[0] != 0:
                    points1 =  np.unique(points1, axis = 1)
                    points[0, 0:len(points1)] = points1
                #print 'after: ', points1
                if points2.shape[0] != 0:                
                    points2 =  np.unique(points2, axis = 1)
                    points[1, 0:len(points2)] = points2
                if points3.shape[0] != 0:   
                    points3 =  np.unique(points3, axis = 1)
                    points[2, 0:len(points3)] = points3
                if points4.shape[0] != 0:     
                    points4 =  np.unique(points4, axis = 1) 
                    points[3, 0:len(points4)] = points4

                labels.append(points)
                image = Image.open(train_file)#.transpose((2,0,1))
                image = transform(image)
                image = np.array(image)
                images.append(image)
            images = np.array(images)
            labels = np.array(labels)
            images = images.transpose((0,3,1,2))
            
            self.dataset_size = images.shape[0] 
        
        return images, labels



def get_tuSimple(train):
    """Get WEBCAM dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([
                                      #transforms.RandomResizedCrop(224),
                                      #transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[92.09105903, 103.22606192, 103.49672837],
                                          std=[48.78478156, 54.54598736, 59.06852698])]
                                          )

    # dataset and data loader
    #print (' dataset and data loader')
    caltech_dataset = OFFICE31(root='/*/*/',
                            sets = "tuSimple",
                            train=train,
                            transform=pre_process,
                            download=False)

    caltech_loader = torch.utils.data.DataLoader(
        dataset=caltech_dataset,
        batch_size=params.batch_size,
        shuffle=True)
    #print(mnist_m_data_loader.data)

    return caltech_loader
