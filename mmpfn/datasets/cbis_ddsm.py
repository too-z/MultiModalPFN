import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from pathlib import Path

from mmpfn.models.dino_v2.models.vision_transformer import vit_base


class CBISDDSMDataset(Dataset):
    def __init__(self, data_path, data_name, kind, image_type='ROI'):
        
        self.kind = kind  # mass calc
        self.data_path = data_path
        self.image_type = image_type # full crop ROI
        
        self.df = pd.read_csv(os.path.join(data_path, data_name))
        # print(self.df[['image file path', 'cropped image file path', 'ROI mask file path']].isna().sum())
        # Define categorical column names explicitly
        if self.kind == 'mass':
            self.cat_cols = ['left or right breast', 'image view', 'abnormality id', 'mass shape', 'mass margins']
            self.numeric_cols = ['breast_density', 'assessment', 'subtlety']
        elif self.kind == 'calc':
            self.cat_cols = ['left or right breast', 'image view', 'abnormality id', 'calc type', 'calc distribution']
            self.numeric_cols = ['breast density', 'assessment', 'subtlety']
            
        col_unused = ['patient_id', 'abnormality type']
        self.target_col = 'pathology'

        self.encoder = OrdinalEncoder()
        self.x = self.encoder.fit_transform(self.df[self.cat_cols])
        self.x = pd.concat([pd.DataFrame(self.x, columns=self.cat_cols), self.df[self.numeric_cols]], axis=1).values
        
        self.target_encoder = LabelEncoder()
        self.df[self.target_col] = self.df[self.target_col].replace('BENIGN_WITHOUT_CALLBACK', 'BENIGN')
        self.y = self.target_encoder.fit_transform(self.df[self.target_col])

    def get_images(self, img_size=14*24): # image size must be a multiple of 14
        
        self.images = []
        
        if self.image_type == 'full':
            image_cols = ['image file path']
        elif self.image_type == 'crop':
            image_cols = ['cropped image file path']
        elif self.image_type == 'ROI':
            image_cols = ['ROI mask file path']
        elif self.image_type == 'all':
            image_cols = ['image file path', 'cropped image file path', 'ROI mask file path']
        
        for i, paths in self.df[image_cols].iterrows():
            image_set = []
            for path in paths:
                image_path = os.path.join(self.data_path, 'jpeg', path.split('/')[-2])
                if not os.path.exists(image_path):
                    print(f"Image {image_path} does not exist, skipping.")
                    continue
                image_path = os.path.join(image_path, os.listdir(image_path)[0])
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    img = np.array(img.resize((img_size, img_size), Image.BILINEAR), dtype=np.float32) 
                    image_set.append(img)
            self.images.append(image_set)
        
        self.images = np.stack(self.images, axis=0)  # (N, H, W, C)
        self.images = torch.from_numpy(
            np.transpose(self.images, (0,1,4,2,3))
        ).float()
        self.images /= 255.0
        
        return self.images
        
    def get_embeddings(self, emb_type='cls', batch_size=1, save=True, mode='train'):
        
        self.batch_size = batch_size
        
        path = f'embeddings/cbis_ddsm/{self.kind}_{emb_type}_{mode}_{self.image_type}.pt'

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
                    batch = self.images[i:i+self.batch_size].to("cuda", non_blocking=True) # Grab a batch of shape [B, 3, H, W, C]
                    batch = batch.view(-1, *batch.shape[2:])  
                    feats = image_encoder.forward_features(batch)
                    if emb_type == 'patch':
                        embs = feats['x_norm_patchtokens']  # [B*3, P, 768]
                    elif emb_type == 'cls':
                        embs = feats['x_norm_clstoken']     # [B*3, 1, 768] or [B*3, 768]
                    else:
                        raise ValueError("Type must be either 'patch' or 'cls'.")

                    if emb_type == 'cls':
                        embs = embs.view(-1, 3, embs.shape[-1])  # Reshape back to [B, 3, 768]
                    else:
                        embs = embs.view(-1, 3 * embs.shape[-2], embs.shape[-1])  # if you keep patches
                    all_embeddings.append(embs.cpu())
                self.embeddings = torch.cat(all_embeddings, dim=0)  # [total_size, 3, 768]
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
