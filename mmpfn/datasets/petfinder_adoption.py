from __future__ import annotations

import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from pathlib import Path

from mmpfn.models.dino_v2.models.vision_transformer import vit_base


class PetfinderAdoptionPredictionDataset(Dataset):
    
    def __init__(
        self, 
        data_path="data/petfinder_adoption",
        is_train=True,
        image_only=False,
    ):
        self.data_path = data_path
        self.image_only = image_only
        self.is_train = is_train
                
        col_features = ["Age","Breed1","Breed2","Color1","Color2","Color3","Dewormed","Fee","FurLength","Gender","Health","MaturitySize","PhotoAmt","State","Sterilized","Type","Vaccinated","VideoAmt","Quantity",]
        col_exclude = ["PetID", "RescureID", "Description", "Name"]
        col_target = "AdoptionSpeed"
        cat_features = ["Breed1","Breed2","Color1","Color2","Color3","Dewormed","FurLength","Gender","Health","MaturitySize","State","Sterilized","Type","Vaccinated",]
        cat_features_index = [col_features.index(feature) for feature in cat_features]
        
        if self.is_train:
            table_path = os.path.join(data_path, "train/train.csv")
            images = [f for f in os.listdir(os.path.join(data_path, "train_images")) if f.endswith(".jpg")]
        else:
            table_path = os.path.join(data_path, "test/test.csv")
            images = [f for f in os.listdir(os.path.join(data_path, "test_images")) if f.endswith(".jpg")]
        
        self.df = pd.read_csv(table_path)
        self.df["PetID"] = self.df["PetID"].astype(str)
        
        images = [f for f in images if f.split("-")[0] in self.df["PetID"].values]
        image_df = pd.DataFrame(
            {
                "PetID": [f.split("-")[0] for f in images],
                "ImageNumber": [f.split("-")[1].split(".")[0] for f in images],
            }
        )
        image_df = image_df[image_df["ImageNumber"] == "1"]
        
        self.df = self.df.merge(image_df, on="PetID", how="left")
        self.df = self.df[self.df["ImageNumber"].notna()]
        self.df["ImagePath"] = self.df["PetID"] + "-1.jpg"
        
        self.y = torch.from_numpy(self.df[col_target].values).float()
        self.x = torch.from_numpy(self.df[col_features].values).float()
        
    def get_images(self, img_size=14*24):
        # image size must be a multiple of 14
        self.images = []
        
        for path in self.df['ImagePath']:
            if self.is_train:
                image_path = os.path.join(self.data_path, "train_images", path)
            else:
                image_path = os.path.join(self.data_path, "test_images", path)
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
    
    def get_embeddings(self, type='patch', batch_size=16, save=True):
        self.batch_size = batch_size
        
        path = f'embeddings/petfinder_adoption/adoption_{type}.pt'

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
                    if type == 'patch':
                        self.embeddings.append(feats['x_norm_patchtokens'].detach().cpu())
                    elif type == 'cls':
                        self.embeddings.append(feats['x_norm_clstoken'].detach().cpu())
                    else:
                        raise ValueError("Type must be either 'patch' or 'cls'.")
                    del batch, feats
                del image_encoder
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
            
            if save:    
                del self.images
                torch.cuda.empty_cache()
                torch.save(self.embeddings, path)    
                
        return self.embeddings

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.x[idx]
        image = self.embeddings[idx//self.batch_size][idx%self.batch_size] if hasattr(self, 'embeddings') else None
        y = self.y[idx]

        return x, image, y

