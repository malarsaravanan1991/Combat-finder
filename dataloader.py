import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from scipy.misc import imread
from torch import Tensor
import cv2
"""
Loads the train/test set. 
Every image in the dataset is 224x224 pixels and the labels are numbered from 0 to 7
for Hit,Kick,Punch,Push,ridehorse,shootgun,stand,wave respectively.
Set root to point to the Train/Test folders.
"""
#root = '/home/malar/prject_vision/'
#rint(dataset)
# Creating a sub class of torch.utils.data.dataset.Dataset
class fight(Dataset):
    def __init__(self, root):
        Image, Y = [], []
        folders = os.listdir(root)
        #value = 0
        for folder in folders:
            folder_path = os.path.join(root, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    Image.append(np.array(imread(img_path)))
                    #print(Img.shape)
                    Y.append(folder) 
                    
                except:
                    print("File {}/{} is broken".format(folder, ims))
            #value = value + 1
        data = [(x, y) for x, y in zip(Image, Y)]
        self.data = data

	# The number of items in the dataset
    def __len__(self):
        return len(self.data)

	# The Dataloader is a generator that repeatedly calls the getitem method.
	# getitem is supposed to return (X, Y) for the specified index.
    def __getitem__(self, index):
        img = self.data[index][0]
        # 8 bit images. Scale between [0,1]. This helps speed up our training
        #r = 224.0 / img.shape[1]
        dim = (224, 224)
        #  perform the actual resizing of the image and show it
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        #img_resized = np.resize(img,(224,224,3))
        img = resized/255.0
        # Input for Conv2D should be Channels x Height x Width
        img_tensor = Tensor(img).view(3, 224, 224).float()
        label = self.data[index][1]
       # label_tensor = torch.FloatTensor(label)
        return (img_tensor, label)
