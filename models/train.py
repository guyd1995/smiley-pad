from argparse import ArgumentParser
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
from itertools import cycle
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
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.dims)
        data = np.array(img) / 255.
        data = torch.Tensor(data)
        data = data.permute(2, 0, 1)
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
        x = F.relu6(x)
        x = self.fc2(x)
        x = x / F.normalize(x, dim=-1)
        return x

    
def batched_index_select(inputs, indices, dim):
    """
    inputs: B x ...
    indices: B (long)
    """
    b, *_ = inputs.shape
    assert inputs.shape[0] == indices.shape[0]
    n = len(inputs.shape)
    unsqueezed_shape = [b] + [1] * (n-1)
    expand_shape = list(inputs.shape)
    expand_shape[dim] = 1
    return inputs.gather(dim=dim, index=indices.reshape(unsqueezed_shape).expand(*expand_shape)).squeeze(dim)
    

def triplet_loss(outputs, target, delta):
    def l2_squared(x): return torch.sum(x**2, dim=-1)
    
    outputs = torch.stack(outputs, dim=1)
    e1 = batched_index_select(outputs, (1+target)%3, dim=1)
    e2 = batched_index_select(outputs, (2+target)%3, dim=1)
    e3 = batched_index_select(outputs, target, dim=1)

    return F.relu(delta + l2_squared(e2-e1) - l2_squared(e3-e1)) + F.relu(delta + l2_squared(e2-e1) - l2_squared(e3-e2))


def train(batch_size, num_steps, lr, device):
    print("load datasets and model..")
    train_ds = FecDataset("FEC_dataset/processed_train.csv")
    test_df = FecDataset("FEC_dataset/processed_test.csv")
    model = FecNet().to(device)

    print("start training")
    delta = .1
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    optimizer = Adam(model.parameters(), lr=lr)
    running_loss = 0
    pbar = tqdm(cycle(train_loader), total=num_steps)
    for i, (inputs, target, triplet_type) in enumerate(pbar):
        inputs = inputs.to(device)
        target = target.to(device)
        triplet_type = triplet_type.to(device)
        
        optimizer.zero_grad()
        inp1, inp2, inp3 = torch.split(inputs, 1, dim=1)
        outputs = list(map(lambda x: model(x.squeeze(1)), (inp1, inp2, inp3)))
        loss = triplet_loss(outputs, target, delta * (1 + triplet_type)).mean()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix_str(f"Loss: {running_loss/(1+i):.2f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", default=30, type=int)
    parser.add_argument("--lr", default=4e-5, type=float)
    parser.add_argument("--num-steps", default=50000, type=int)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(batch_size=args.batch_size, num_steps=args.num_steps, lr=args.lr, device=device)
