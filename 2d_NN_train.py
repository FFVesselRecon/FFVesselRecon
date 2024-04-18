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


train = BiPlane(csv_file = "D:/ALL_GROUND_TRUTH_VOLUMES/train_predict_2D_slice.csv") 
test = BiPlane(csv_file ="D:/ALL_GROUND_TRUTH_VOLUMES/test_predict_2D_slice.csv")

batch_size = 64

train_loader = DataLoader(dataset = train, batch_size = batch_size,shuffle = True)
test_loader =DataLoader(dataset =  test, batch_size =batch_size,shuffle = False)


print(train.__getitem__(9128))


# In[5]:


import gc

gc.collect()
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_cached())


# In[6]:


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


# In[7]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


model = Generator(768, 1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scaler = GradScaler()

SAVE_PATH = 'D:/ALL_GROUND_TRUTH_VOLUMES/2D_prediction'
os.makedirs(SAVE_PATH, exist_ok=True)  
checkpoint_epoch = 1000000
checkpoint_path = os.path.join(SAVE_PATH, f"model_epoch_{checkpoint_epoch}.pt")

if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)  
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")
else:
    print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
    checkpoint_epoch = 0  

num_epochs = 100



for epoch in range(num_epochs):
    model.train()
    cumulative_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}, Training Loss = 0.0000")
    
    for i, data in enumerate(progress_bar):
        vec1, vec2, image = data 
        vec1, vec2, image = vec1.to(device), vec2.to(device), image.to(device)
        
        combined_vec = torch.cat((vec1, vec2), dim=1)
     
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(combined_vec.float())
            loss = criterion(outputs.squeeze(1), image)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        cumulative_loss += loss.item()
        avg_loss = cumulative_loss / (i + 1)
        progress_bar.set_description(f"Epoch {epoch+1}, Training Loss = {avg_loss:.7f}")
        
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




