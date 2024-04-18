#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import gc

gc.collect()
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_cached())


# In[3]:


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



# In[4]:


#train = BiPlane(csv_file = "C:/Users/nirc/Downloads/overfit two example - overfit one example.csv") 

train = BiPlane(csv_file = "D:/ALL_GROUND_TRUTH_VOLUMES/train_predict_2D_slice.csv") 
test = BiPlane(csv_file ="D:/ALL_GROUND_TRUTH_VOLUMES/test_predict_2D_slice.csv")

batch_size = 16

train_loader = DataLoader(dataset = train, batch_size = batch_size,shuffle = True)
test_loader =DataLoader(dataset =  test, batch_size =batch_size,shuffle = False)


print(train.__getitem__(0))


# In[5]:


import gc

gc.collect()
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_cached())


# In[6]:


import torch
import numpy as np

class GaussianFourierFeatureTransform(torch.nn.Module):

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        torch.manual_seed(42) 
        self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)
        x = x @ self._B.to(x.device)
        x = x.view(batches, width, height, self._mapping_size)
        x = x.permute(0, 3, 1, 2)
        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)
coords = np.linspace(0, 1, 384, endpoint=False)
xy_grid = np.stack(np.meshgrid(coords, coords), -1)
xy_grid = torch.tensor(xy_grid).float().contiguous()
xy_grid = xy_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1).permute(0, 3, 1, 2)

print(xy_grid.shape)

torch.manual_seed(42) 
fourier_features = GaussianFourierFeatureTransform(2, 128, 10).cuda()
pos_enc = fourier_features(xy_grid).cuda()

print(pos_enc.shape)


# In[7]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class DualEncoder(nn.Module):
    def __init__(self):
        super(DualEncoder, self).__init__()
        self.fc1 = nn.Linear(768, 128) 
        self.fc2 = nn.Linear(128, 32 * 48 * 48)  
        
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
      
        self.upsample = nn.ConvTranspose2d(64, 64, kernel_size=16, stride=8, padding=4, output_padding=0)
        
        self.final_conv = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, input1, input2):
        x1 = F.relu(self.fc1(input1))
        x1 = F.relu(self.fc2(x1))
        x1 = x1.view(-1, 32, 48, 48) 
        x1 = self.relu(self.bn1(self.conv1(x1)))
        
        x1 = self.upsample(x1)  
        
        x2 = self.encoder2(input2)  
        x = torch.cat((x1, x2), dim=1)  
        
        x = self.final_conv(x)  
        return torch.sigmoid(x)  



# In[ ]:


import os
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

model = DualEncoder()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scaler = GradScaler()

SAVE_PATH = 'D:/ALL_GROUND_TRUTH_VOLUMES/transformer_cp'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

checkpoint_epoch = 100000
checkpoint_path = os.path.join(SAVE_PATH, f"model_epoch_{checkpoint_epoch}.pt")

if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    checkpoint_epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from {checkpoint_path}")
else:
    print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
    checkpoint_epoch = 0

num_epochs = 1000


for epoch in range(checkpoint_epoch, num_epochs):
    model.train()
    cumulative_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}, Training Loss = 0.0000")
    
    for i, data in enumerate(progress_bar):
        vec1, vec2, image = data
        vec1, vec2, image = vec1.to(device), vec2.to(device), image.to(device)
        
        combined_vec = torch.cat((vec1, vec2), dim=1)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(combined_vec,pos_enc)  
            loss = criterion(outputs.squeeze(1), image)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        cumulative_loss += loss.item()
        avg_loss = cumulative_loss / (i + 1)
        progress_bar.set_description(f"Epoch {epoch+1}, Training Loss = {avg_loss:.7f}")
    
    print(f"Epoch {epoch+1}, Training Loss = {avg_loss:.4f}")
    
    if (epoch + 1) % 1 == 0:
        checkpoint_file = os.path.join(SAVE_PATH, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_file)
        print(f"Saved model parameters to {checkpoint_file}")

print("Training finished.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




