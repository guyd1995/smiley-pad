import torchvision
from torchvision.models import DenseNet
from torchvision.models.densenet import _DenseBlock
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import importlib
facenet_pytorch = importlib.import_module('facenet-pytorch')
InceptionResnetV1 = facenet_pytorch.models.inception_resnet_v1.InceptionResnetV1


def intermediate_layer_getter(model, layer_name, register_inplace=False):
    if not register_inplace:
        model = deepcopy(model)
    
    class _IntermediateLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._out = None
            self.model = model
            layer = model.get_submodule(layer_name)
            self._hooks = [layer.register_forward_hook(self._remember_hook)]
        
        def _remember_hook(self, mod, inp, outp):
            self._out = outp
            return outp
        
        def forward(self, *args, **kwargs):
            self.model(*args, **kwargs)
            return self._out
        
    return _IntermediateLayerModel()


class FecDataset(Dataset):
    def __init__(self):
        pass
    
    def __getitem__(self, idx):
        pass
    
    def __len__(self):
        pass


class FecNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = InceptionResnetV1(pretrained='vggface2').eval()        
        self.backbone = intermediate_layer_getter(resnet, 'mixed_6a') 
        self.conv = nn.Conv2d(896, 512, (1,1))
        self.dense_block = _DenseBlock(num_layers=5, num_input_features=512, growth_rate=64, bn_size=4, drop_rate=0)
        self.avg_pool = nn.AvgPool2d(12)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(832, 512)
        self.fc2 = nn.Linear(512, 16)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        x = F.relu(x)
        x = self.dense_block(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        x = x / F.normalize(x, dim=-1)
        return x
    

def triplet_loss(outputs, target, delta):
    indices = [((0, 1), (1, 2), (0, 2)), ((1, 2), (0, 1), (0, 2)), ((0, 2), (0, 1), (1, 2))]
    good, bad1, bad2 = map(lambda idx: F.sum((outputs[idx[1]]-outputs[idx[0]])**2, dim=-1), indices[target])
    return F.relu(delta + good - bad1) + F.relu(delta + good - bad2)

def train():
    train_ds = FecDataset()
    model = FecNet()

    lr = 1e-3
    batch_size = 8
    train_loader = DataLoader(train_ds, batch_size=8)
    optimizer = Adam(model.parameters(), lr=lr)
    for inputs, target in tqdm(train_loader):
        optimizer.zero_grad()
        *outputs = map(model, inputs)
        n_classes = 1
        loss = triplet_loss(outputs, target, delta * n_classes)
        loss.backward()
        optimizer.step()