import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from pathlib import Path

from mmpfn.models.dino_v2.models.vision_transformer import vit_base


class PawpularityDataset(Dataset):
    def __init__(self, data_path):
        
        self.data_path = data_path
        
        self.df = pd.read_csv(os.path.join(data_path, "train.csv"))
        
        self.target_col = 'Pawpularity'
        self.cat_cols = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']
        self.image_cols = ['Id']

        self.encoder = OrdinalEncoder()
        self.x = self.encoder.fit_transform(self.df[self.cat_cols])
        
        # self.target_encoder = LabelEncoder()
        # self.y = self.target_encoder.fit_transform(self.df[self.target_col])
        self.y = self.df[self.target_col].values / 100.0  # normalize to [0, 1]
        

    def get_images(self, img_size=14*24):
        # image size must be a multiple of 14
        self.images = []
        for _, paths in self.df[self.image_cols].iterrows():
            for path in paths:
                image_path = os.path.join(self.data_path, "train", path + ".jpg")
                if not os.path.exists(image_path):
                    print(f"Image {image_path} does not exist, skipping.")
                    continue
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    img = np.array(img.resize((img_size, img_size), Image.BILINEAR), dtype=np.float32)
                    self.images.append(img)
        self.images = np.stack(self.images, axis=0)  # (N, H, W, C)
        self.images = torch.from_numpy(
            np.transpose(self.images, (0,3,1,2))
        ).float()

        self.images /= 255.0
        
        return self.images
        
    def get_embeddings(self, batch_size=16, save=True):
        
        self.batch_size = batch_size
        
        path = f'embeddings/pawpularity/pawpularity.pt'

        if os.path.exists(path):
            print(f"Load embeddings from {path}")
            self.embeddings = torch.load(path)
        else:
            image_encoder = vit_base(
                patch_size=14, img_size=518, init_values=1.0, num_register_tokens=0, block_chunks=0
            )

            image_model_path = f"{Path().absolute()}/parameters/dinov2_vitb14_pretrain.pth"
            image_state_dict = torch.load(image_model_path)
            image_encoder.load_state_dict(image_state_dict)
            _ = image_encoder.cuda().eval()

            self.embeddings = []

            with torch.no_grad():
                all_embeddings = []
                for i in range(0, self.images.shape[0], self.batch_size):
                    batch = self.images[i:i+self.batch_size].to("cuda", non_blocking=True)
                    batch = batch.view(-1, *batch.shape[1:])  
                    feats = image_encoder.forward_features(batch)
                    
                    embs = feats['x_norm_clstoken']
                    embs = embs.view(-1, len(self.image_cols), embs.shape[-1])
                    
                    all_embeddings.append(embs.cpu())
                self.embeddings = torch.cat(all_embeddings, dim=0)
            if save:
                torch.save(self.embeddings, path)    
        
        return self.embeddings
            

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.x[idx]
        image = self.embeddings[idx//self.batch_size] if hasattr(self, 'embeddings') else None
        y = self.y[idx]

        return x, image, y
