import os
import torch
import numpy as np
import pandas as pd

import h5py

from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from pathlib import Path

from mmpfn.models.dino_v2.models.vision_transformer import vit_base


class OL3IDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        
        self.data_path = data_path
        self.mode = mode
        
        self.df = pd.read_csv(os.path.join(data_path, 'clinical_data.csv'))

        self.cat_cols, self.num_cols = [], []
        for col in self.df.columns:
            if col in ['anon_id', 'label_1y', 'set_1y', 'label_5y', 'set_5y']:
                continue
            if self.df[col].dtype == 'object':
                self.cat_cols.append(col)
            else:
                if self.df[col].nunique() < 4:
                    self.cat_cols.append(col)
                else:
                    self.num_cols.append(col)

        self.target_col = 'label_1y' # label_5y

        if mode == 'train':
            self.df = self.df[(self.df['set_1y']=='train') | (self.df['set_1y']=='val')].reset_index(drop=True)
        elif mode == 'valid':
            self.df = self.df[self.df['set_1y']=='val'].reset_index(drop=True)
        elif mode == 'trainonly':
            self.df = self.df[self.df['set_1y']=='train'].reset_index(drop=True)
        else:
            self.df = self.df[self.df['set_1y']=='test'].reset_index(drop=True)

        self.encoder = OrdinalEncoder()
        self.x = self.encoder.fit_transform(self.df[self.cat_cols].astype(str))
        self.x = pd.concat([pd.DataFrame(self.x, columns=self.cat_cols), self.df[self.num_cols]], axis=1).values
        
        self.target_encoder = LabelEncoder()
        self.y = self.target_encoder.fit_transform(self.df[self.target_col])

    def get_images(self, img_size=14*36):
        # image size must be a multiple of 14
        self.images = []
        file_name = 'l3_slices.h5'
        
        with h5py.File(os.path.join(self.data_path, file_name), 'r') as f:
            for id in self.df['anon_id']:
                img = Image.fromarray(f[id][()])
                img = img.convert('RGB')
                img = np.array(img.resize((img_size, img_size), Image.BILINEAR), dtype=np.float32)
                self.images.append(img)
        
        self.images = np.stack(self.images, axis=0)  # (N, H, W, C)
        self.images = torch.from_numpy(
            np.transpose(self.images, (0,3,1,2))
        ).float()

        self.images /= 255.0
        
        return self.images
        
    def get_embeddings(self, image_type, batch_size=16, save=True):
        
        self.batch_size = batch_size
        
        path = f'embeddings/ol3i/{image_type}_{self.mode}.pt'

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
                for i in range(0, self.images.shape[0], self.batch_size):
                    batch = self.images[i:i+self.batch_size].to("cuda", non_blocking=True)
                    feats = image_encoder.forward_features(batch)
                    if image_type == 'patch':
                        self.embeddings.append(feats['x_norm_patchtokens'].detach().cpu())
                    elif image_type == 'cls':
                        self.embeddings.append(feats['x_norm_clstoken'].detach().cpu())
                    else:
                        raise ValueError("Type must be either 'patch' or 'cls'.")

            torch.cuda.empty_cache()
            print(f"Embeddings shape: {self.embeddings[0].shape}", len(self.embeddings))
            # Suppose self.embeddings is a list of tensors [B_i, ...] along dim=0
            sizes = [t.size(0) for t in self.embeddings]
            total_size = sum(sizes)

            # Preallocate
            final_shape = (total_size, *self.embeddings[0].shape[1:])
            out = torch.empty(final_shape, dtype=self.embeddings[0].dtype, device=self.embeddings[0].device)

            # Copy in place
            offset = 0
            for t in self.embeddings:
                out[offset:offset + t.size(0)] = t
                offset += t.size(0)

            self.embeddings = out

            print(f"Embeddings shape: {self.embeddings.shape}")
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
