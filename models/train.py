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


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

    
def get_model(path, device='cpu'):
    model = FecNet().to(device)
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=torch.device(device))['state_dict'])
    return model


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


IMG_DIMS = (224, 224)
def preprocess_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_DIMS)
    data = (np.array(img) - 127.5) / 128. # the pre-processing method from facenet-pytorch
    data = torch.Tensor(data)
    data = data.permute(2, 0, 1)
    return data


class FecDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        img0, img1, img2 = map(preprocess_img, (data['img0'], data['img1'], data['img2']))
        inputs = torch.stack([img0, img1, img2], dim=0)
        target = data['target']
        triplet_type = data['triplet_type']
        return inputs, torch.LongTensor([target]), torch.LongTensor([triplet_type])
            
    def __len__(self):
        return len(self.data)


class FecNet(nn.Module):
    def __init__(self, dropout_rate=.5):
        super().__init__()
        self.facenet = self._get_truncated_facenet()
        self.densenet = DenseNet(block_config=(5,), num_init_features=512, 
                                 growth_rate=64, num_classes=16, 
                                 drop_rate=dropout_rate)
        self.densenet.features[0] = nn.Conv2d(1792, 512, kernel_size=7, stride=2, padding=3, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
    
    @staticmethod
    def _get_truncated_facenet():
        facenet = InceptionResnetV1(pretrained='vggface2', classify=True).eval()
        facenet.avgpool_1a = Identity()
        facenet.last_linear = Identity()
        facenet.last_bn = Identity()
        facenet.logits = Identity()
        facenet = intermediate_layer_getter(facenet, 'mixed_7a')
        facenet.requires_grad_(False)
        return facenet
    
    def forward(self, x):
        x = self.facenet(x)
#         x = x.reshape(x.shape[0], 1792, 5, 5)
        x = self.densenet(x)
        x = F.normalize(x, p=2, dim=-1)
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
    

def get_metrics(outputs, target, delta):
    def l2_squared(x): return torch.sum(x**2, dim=-1)
    
    outputs = torch.stack(outputs, dim=1)
    e1 = batched_index_select(outputs, (1+target)%3, dim=1)
    e2 = batched_index_select(outputs, (2+target)%3, dim=1)
    e3 = batched_index_select(outputs, target, dim=1)
    L12 = l2_squared(e2-e1)
    L13 = l2_squared(e3-e1)
    L23 = l2_squared(e3-e2)
    accuracy = (L12 < L13) * (L12 < L23)
    loss = F.relu(delta + L12 - L13) + F.relu(delta + L12 - L23)
    return loss.mean(), accuracy.float().mean()


def train(batch_size, num_steps, lr, device, checkpoint_folder, checkpoint_freq, checkpoint_model):
    print("load datasets and model..")
    train_ds = FecDataset("../../FEC_dataset/processed_train.csv")
    test_df = FecDataset("../../FEC_dataset/processed_test.csv")
    model = get_model(path=checkpoint_model, device=device)
    print("start training")
    delta = .1
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr)
    running_loss = 0
    running_acc = 0
    pbar = tqdm(enumerate(train_loader), total=num_steps)
    for i, (inputs, target, triplet_type) in pbar:
        if i >= num_steps:
            break
        inputs = inputs.to(device)
        target = target.to(device)
        triplet_type = triplet_type.to(device)
        
        optimizer.zero_grad()
        inp1, inp2, inp3 = torch.split(inputs, 1, dim=1)
        outputs = list(map(lambda x: model(x.squeeze(1)), (inp1, inp2, inp3)))
        loss, accuracy = get_metrics(outputs, target, delta * (1 + triplet_type))
        
        loss.backward()
        optimizer.step()
        running_acc += accuracy.item()
        running_loss += loss.item()
        avg_loss = running_loss / (1+i)
        avg_acc = running_acc / (1+i)
        pbar.set_postfix_str(f"Loss: {avg_loss:.2f} Acc: {avg_acc:.2f}")
        
        if (1 + i) % checkpoint_freq == 0:
            torch.save({"num_steps": num_steps, 
                        "state_dict": model.state_dict(), 
                        "loss": avg_loss},
                      f"{checkpoint_folder}/model.pt")


def evaluate(batch_size, num_steps, device, checkpoint_model):
    print("load datasets and model..")
    test_ds = FecDataset("../../FEC_dataset/processed_test.csv")
    model = get_model(path=checkpoint_model, device=device)
    model.eval()
    print("start evaluation..")
    delta = .1
    test_loader = iter(DataLoader(test_ds, batch_size=batch_size, shuffle=True))
    running_loss = 0
    running_acc = 0
    pbar = tqdm(range(num_steps), total=num_steps)
    i = 0
    for _ in pbar:
        try: 
            (inputs, target, triplet_type) = next(test_loader)
        except:
            print("error")
            continue
        inputs = inputs.to(device)
        target = target.to(device)
        triplet_type = triplet_type.to(device)
        
        inp1, inp2, inp3 = torch.split(inputs, 1, dim=1)
        outputs = list(map(lambda x: model(x.squeeze(1)), (inp1, inp2, inp3)))
        loss, accuracy = get_metrics(outputs, target, delta * (1 + triplet_type))
        
        running_acc += accuracy.item()
        running_loss += loss.item()
        avg_loss = running_loss / (1+i)
        avg_acc = running_acc / (1+i)
        pbar.set_postfix_str(f"Loss: {avg_loss:.2f} Acc: {avg_acc:.2f}")
        i += 1

        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", default=30, type=int)
    parser.add_argument("--lr", default=4e-5, type=float)
    parser.add_argument("--num-steps", default=50000, type=int)
    parser.add_argument("--checkpoint-freq", default=500, type=int)
    parser.add_argument("--from-checkpoint", default=None, type=str)
    parser.add_argument("--do_eval", action="store_true")
    args = parser.parse_args()
    checkpoint_folder = "checkpoints"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.do_eval:
        evaluate(batch_size=args.batch_size, num_steps=args.num_steps, device=device, 
                 checkpoint_model=args.from_checkpoint)        
    else:
        train(batch_size=args.batch_size, num_steps=args.num_steps, lr=args.lr, 
              checkpoint_folder=checkpoint_folder,
              checkpoint_freq=args.checkpoint_freq, device=device, checkpoint_model=args.from_checkpoint)
