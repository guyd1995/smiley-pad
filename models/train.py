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
from itertools import cycle


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
    dims = (224, 224)
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        img0, img1, img2 = map(self._preprocess_img, (data['img0'], data['img1'], data['img2']))
        inputs = torch.stack([img0, img1, img2], dim=0)
        target = data['target']
        triplet_type = data['triplet_type']
        return inputs, torch.LongTensor([target]), torch.LongTensor([triplet_type])
        
    def _preprocess_img(self, img_path):
        img = Image.open(img_path)
        img = img.resize(self.dims)
        data = np.array(img) / 255.
        data = torch.Tensor(data)
        data = data.transpose(1, 2, 0)
        return data
    
    def __len__(self):
        return len(self.data)


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
    # TODO: batch this
    indices = [((1, 2), (0, 1), (0, 2)), ((0, 2), (0, 1), (1, 2)), ((0, 1), (1, 2), (0, 2))]
    good, bad1, bad2 = map(lambda idx: F.sum((outputs[idx[1]]-outputs[idx[0]])**2, dim=-1), indices[target])
    return F.relu(delta + good - bad1) + F.relu(delta + good - bad2)

def train():
    train_ds = FecDataset()
    model = FecNet()

    lr = 5e-4
    num_steps = 50000
    batch_size = 30
    delta = .1
    train_loader = DataLoader(train_ds, batch_size=8)
    optimizer = Adam(model.parameters(), lr=lr)
    for inputs, target, triplet_type in tqdm(cycle(train_loader), total=num_steps):
        optimizer.zero_grad()
        inp1, inp2, inp3 = torch.split(inputs, 1, dim=1)
        *outputs = map(model, (inp1, inp2, inp3))
        loss = triplet_loss(outputs, target, delta * (1 + triplet_type))
        loss.backward()
        optimizer.step()

        
if __name__ == "__main__":
    train()