#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import torchvision.transforms as transforms
import time
import random
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
import os
import pandas as pd 
import numpy as np   
import cv2          
import matplotlib.pyplot as plt  
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, datasets, models, utils
from torch.utils.data import Dataset, DataLoader 
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from skimage import io, transform
from torch.optim import lr_scheduler
from skimage.transform import AffineTransform, warp
from tqdm import tqdm
device = torch.device('cuda')
import albumentations as A
from torch.cuda.amp import autocast, GradScaler

import os
import time
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from matplotlib import pyplot as plt
import numpy as np


# In[ ]:


import gc

gc.collect()
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_cached())


# In[ ]:


import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import torchvision.transforms as transforms

class BiPlane(Dataset):
    def __init__(self, csv_file, transform=None, augment=False):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        
        vector1 = self.annotations.iloc[index, 0]
        vector2 = self.annotations.iloc[index, 1]

        
        
        list_elements_1 = vector1.strip('[]').split(',')
        list_elements_1 = [int(elem.strip()) for elem in list_elements_1]
        numpy_array_1 = np.array(list_elements_1)
        vector1 = torch.tensor(numpy_array_1)

        
        list_elements_2 = vector2.strip('[]').split(',')
        list_elements_2 = [int(elem.strip()) for elem in list_elements_2]
        numpy_array_2 = np.array(list_elements_2)
        vector2 = torch.tensor(numpy_array_2)
      
        
        
        
        
        
        
        img = cv2.imread(self.annotations.iloc[index,2],0)
        
        img = torch.tensor(img)

        img = img.float() / 255.0
       


        return vector1/255.0,vector2/255.0,img



# In[ ]:


train = BiPlane(csv_file = "D:/ALL_GROUND_TRUTH_VOLUMES/train_predict_2D_slice.csv") 
test = BiPlane(csv_file ="D:/ALL_GROUND_TRUTH_VOLUMES/test_predict_2D_slice.csv")

batch_size = 1

train_loader = DataLoader(dataset = train, batch_size = batch_size,shuffle =False)
test_loader =DataLoader(dataset =  test, batch_size =batch_size,shuffle = False)


print(train.__getitem__(9128))


# In[ ]:


import gc

gc.collect()
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_cached())


# In[ ]:


import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, output_channels):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_dim, 256 * 24 * 24) 
        
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  
        self.conv4 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)  
        
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 24, 24) 
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.sigmoid(self.conv4(x)) 
        return x

input_dim = 384*2
output_channels = 1

model = Generator(input_dim, output_channels)
input_vector = torch.randn(2, input_dim)  
print(input_vector.shape)
output_image = model(input_vector)
print(output_image.shape) 


# In[ ]:


import torch
import numpy as np
from torch.cuda.amp import autocast
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

file_path = 'D:/ALL_GROUND_TRUTH_VOLUMES/2D_prediction/model_epoch_60.pt'
checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Generator(768, 1)  
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

progress_bar = tqdm(test_loader, desc="Testing")
outputs_list = []
volume_counter = 0

test_volumes = [0, 1, 13, 32]

psnr_list = []
ssim_list = []
mse_list = []
mae_list = []

with torch.no_grad():
    for i, data in enumerate(progress_bar):
        vec1, vec2, image = data
        vec1 = vec1.to(device)
        vec2 = vec2.to(device)
        image = image.to(device)

        combined_vec = torch.cat((vec1, vec2), dim=1)

        with autocast():
            outputs = model(combined_vec.float()).squeeze(0).squeeze(0).detach().cpu().numpy()
            outputs_list.append(outputs)
            
            if len(outputs_list) == 384:
                volume = np.stack(outputs_list, axis=0)
                volume_filename = f'D:/ALL_GROUND_TRUTH_VOLUMES/test/{test_volumes[volume_counter]}/stacked_volume.npy'
                np.save(volume_filename, volume)

                gt_volume_filename = f'D:/ALL_GROUND_TRUTH_VOLUMES/test/{test_volumes[volume_counter]}/thresh3D_32.npy'
                gt_volume = np.load(gt_volume_filename)

                gt_volume = (gt_volume - gt_volume.min()) / (gt_volume.max() - gt_volume.min())

                volume_psnr = psnr(gt_volume, volume, data_range=gt_volume.max() - gt_volume.min())
                volume_ssim = ssim(gt_volume, volume, data_range=gt_volume.max() - gt_volume.min())
                volume_mse = np.mean((gt_volume - volume) ** 2)
                volume_mae = np.mean(np.abs(gt_volume - volume))

                psnr_list.append(volume_psnr)
                ssim_list.append(volume_ssim)
                mse_list.append(volume_mse)
                mae_list.append(volume_mae)

                volume_counter += 1
                outputs_list = []

overall_psnr, std_psnr = np.mean(psnr_list), np.std(psnr_list)
overall_ssim, std_ssim = np.mean(ssim_list), np.std(ssim_list)
overall_mse, std_mse = np.mean(mse_list), np.std(mse_list)
overall_mae, std_mae = np.mean(mae_list), np.std(mae_list)

print(f"Overall Metrics with Error Bars:")
print(f"PSNR: {overall_psnr} ± {std_psnr}, SSIM: {overall_ssim} ± {std_ssim}, MSE: {overall_mse} ± {std_mse}, MAE: {overall_mae} ± {std_mae}")


# In[ ]:




