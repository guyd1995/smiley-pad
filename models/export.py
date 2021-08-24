from train import FecNet, preprocess_img
import json
from argparse import ArgumentParser
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch import nn
import torch.nn.functional as F
from PIL import Image


def add_emoji_classifier(model, emoji_def):
    model.eval()
    all_samples = []
    for i, emoji_data in enumerate(emoji_def):
        samples = emoji_data['samples']
        all_samples.extend(samples)
    
    sample_vectors = []
    with torch.no_grad():
        for img_path in all_samples:
            sample_vectors.append(model(preprocess_img(img_path).unsqueeze(0)).squeeze(0).detach())
        sample_vectors = torch.stack(sample_vectors, dim=1)
        
    class EmojiClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.sample_vectors = sample_vectors
            
        def forward(self, x):
            x = self.model(x)           # (B, embed_dim)
            x = x @ self.sample_vectors # (B, n_samples)
            return x

    return EmojiClassifier()


def export(model, example_shape = (1, 3, 224, 224), mobile_model_name="face_model.ptl"):
    example = torch.rand(*example_shape)
    model.eval()
    traced_model = torch.jit.trace(model, example)
    mobile_model = optimize_for_mobile(traced_model)
    mobile_model._save_for_lite_interpreter(mobile_model_name)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--emoji-json", type=str, default="emoji_def.json")
    parser.add_argument("--to-assets-folder", action="store_true")
    args = parser.parse_args()
    
    model_path = args.model
    emoji_json_path = args.emoji_json
    to_assets = args.to_assets_folder
    
    print("fetching model and json")
    with open(emoji_json_path, "r") as f:
        emoji_def = json.load(f)
    
    model = FecNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'], strict=False)
    model = add_emoji_classifier(model, emoji_def)
    print("exporting model and json")
    export(model)
    idx2emoji = list(map(lambda x: x['value'], emoji_def))
    with open("idx2emoji.json", "w") as f:
        json.dump(f, idx2emoji)
    
    if to_assets:
        raise NotImplementedError