import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from pathlib import Path

from mmpfn.models.dino_v2.models.vision_transformer import vit_base


class PADUFES20Dataset(Dataset):
    def __init__(self, data_path):
        
        self.data_path = data_path
        
        self.df = pd.read_csv(os.path.join(data_path, "metadata.csv"))
        
        # Define categorical column names explicitly
        self.bool_cats = ['smoke', 'drink', 'pesticide', 'skin_cancer_history', 'cancer_history','has_piped_water', 'has_sewage_system', 'itch', 'grew', 'hurt','bleed', 'elevation', 'biopsed', 'changed']
        self.string_cats = ['background_father', 'background_mother', 'gender', 'region']
        self.numeric_cols = ['age', 'diameter_1', 'diameter_2']
        self.target_col = 'diagnostic'
        # self.drop_cols = ['patient_id', 'lesion_id', 'img_id']
        self.cat_cols = self.bool_cats + self.string_cats

        self.encoder = OrdinalEncoder()
        self.x = self.encoder.fit_transform(self.df[self.cat_cols])
        self.x = pd.concat([pd.DataFrame(self.x, columns=self.cat_cols), self.df[self.numeric_cols]], axis=1).values
        
        self.target_encoder = LabelEncoder()
        self.y = self.target_encoder.fit_transform(self.df[self.target_col])

    def get_images(self, img_size=14*24):
        # image size must be a multiple of 14
        self.images = []
        
        for path in self.df['img_id']:
            image_path = os.path.join(self.data_path, "imgs", path)
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
        
        path = f'image_embeddings/pad_ufes_20/pad_ufes_20_{type}.pt'

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
