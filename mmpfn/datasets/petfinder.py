from __future__ import annotations

import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from pathlib import Path

from mmpfn.models.dino_v2.models.vision_transformer import vit_base

from transformers import AutoTokenizer, AutoModel


class PetfinderDataset(Dataset):
    
    def __init__(
        self, 
        data_path="data/petfinder_adoption",
        is_train=True,
        image_only=False,
    ):
        self.data_path = data_path
        self.image_only = image_only
        self.is_train = is_train
                
        col_features = ["Breed1","Breed2","Color1","Color2","Color3","Dewormed","FurLength","Gender","Health","MaturitySize","State","Sterilized","Type","Vaccinated","Age","VideoAmt","Quantity","PhotoAmt","Fee",]
        col_exclude = ["PetID", "RescureID", "Name"]
        text_features = ["Description"]
        col_target = "AdoptionSpeed"
        self.cat_features = ["Breed1","Breed2","Color1","Color2","Color3","Dewormed","FurLength","Gender","Health","MaturitySize","State","Sterilized","Type","Vaccinated",]
        num_features = list(set(col_features) - set(self.cat_features))
        
        table_path = os.path.join(data_path, "train/train.csv")
        images = [f for f in os.listdir(os.path.join(data_path, "train_images")) if f.endswith(".jpg")]
        
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
        
        self.image_features = "ImagePath"
        self.df = self.df.merge(image_df, on="PetID", how="left")
        self.df = self.df[self.df["ImageNumber"].notna()]
        self.df[self.image_features] = self.df["PetID"] + "-1.jpg"
        self.df = self.df[self.df[self.image_features].notna()]
        
        self.text = self.df[text_features]
        self.text.loc[self.text["Description"].isnull(),"Description"] = ''
        
        self.target_encoder = LabelEncoder()
        self.y = self.target_encoder.fit_transform(self.df[col_target])
        
        # self.x = torch.from_numpy(self.df[col_features].values).float()
        self.encoder = OrdinalEncoder()
        self.x = self.encoder.fit_transform(self.df[self.cat_features])
        self.x = pd.concat([pd.DataFrame(self.x, columns=self.cat_features), self.df[num_features]], axis=1).values
        
    def get_images(self, img_size=14*24):
        # image size must be a multiple of 14
        self.images = []
        
        for i, paths in self.df[[self.image_features]].iterrows():
            image_set = []
            for path in paths:
                image_path = os.path.join(self.data_path, 'train_images', path)
                if not os.path.exists(image_path):
                    print(f"Image {image_path} does not exist, skipping.")
                    continue
                # image_path = os.path.join(image_path, os.listdir(image_path)[0])
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    # img = np.array(img.resize((img_size, img_size), Image.BILINEAR), dtype=np.float3) 
                    img = np.array(img.resize((img_size, img_size), Image.BILINEAR), dtype=np.uint8) 
                    image_set.append(img)
            self.images.append(image_set)
            # if i > 10:
            #     break
        
        self.images = np.stack(self.images, axis=0)  # (B, N, H, W, C)
        self.images = torch.from_numpy(np.transpose(self.images, (0,1,4,2,3))).float() # (B, N, C, H, W)
        self.images /= 255.0
        
        return self.images
    
    def get_embeddings(
        self, 
        batch_size=16, 
        multimodal_type='all' # image, text
    ):
        model_name = 'dinov2'
        # model_name = 'dinov3'
        path = f'embeddings/petfinder/petfinder_{multimodal_type}_{model_name}.pt'

        if os.path.exists(path):
            print(f"Load embeddings from {path}")
            self.embeddings = torch.load(path)
        else:
            local_image = True
            if multimodal_type == 'image' or multimodal_type == 'all':
                if local_image:
                    image_encoder = vit_base(patch_size=14, img_size=518, init_values=1.0, num_register_tokens=0, block_chunks=0)
                    image_model_path = f"{Path().absolute()}/parameters/dinov2_vitb14_pretrain.pth"
                    image_state_dict = torch.load(image_model_path)
                    image_encoder.load_state_dict(image_state_dict)
                    _ = image_encoder.cuda().eval()
                else:
                    MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
                    image_encoder = AutoModel.from_pretrained(MODEL_ID).cuda().eval()

                self.embeddings_image = []
                with torch.no_grad():
                    all_image_embeddings = []
                    for i in range(0, self.images.shape[0], batch_size):
                        batch = self.images[i:i+batch_size].to("cuda", non_blocking=True) # Grab a batch of shape [B, N, H, W, C]
                        batch = batch.view(-1, *batch.shape[2:])  
                        
                        if local_image:
                            feats = image_encoder.forward_features(batch)
                            embs = feats['x_norm_clstoken']
                        else:
                            feats = image_encoder(batch)
                            embs = feats['last_hidden_state'][:,0,:]
                        
                        embs = embs.view(-1, self.images.shape[1], embs.shape[-1])  # Reshape back to [B, N, 768]
                        all_image_embeddings.append(embs.cpu())
                        
                    torch.cuda.empty_cache()
                    self.embeddings_image = torch.cat(all_image_embeddings, dim=0).cpu()  # [total_size, N, 768]
                torch.cuda.empty_cache()
                
                if multimodal_type == 'image':
                    self.embeddings = self.embeddings_image
                    torch.cuda.empty_cache()
                    torch.save(self.embeddings, path)       
                    return self.embeddings
                
            if multimodal_type == 'text' or multimodal_type == 'all':
                local_text = True
                # model_name = "microsoft/deberta-v3-base" 
                # local_dir = "datasets/deberta"
                model_name = "google/electra-base-discriminator"
                local_dir = "models/electra"
                if 'deberta' in model_name:
                    use_fast = False
                else:
                    use_fast = True
                
                if local_text:
                    tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=use_fast, local_files_only=True)
                    text_encoder = AutoModel.from_pretrained(local_dir, local_files_only=True).cuda().eval()
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
                    text_encoder = AutoModel.from_pretrained(model_name).cuda().eval()

                self.embeddings_text = []
                with torch.no_grad():
                    for i, texts in tqdm(self.text.iterrows()):
                        last_hidden_states = []
                        for text in texts:
                            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                            inputs = {key: value.to('cuda') for key, value in inputs.items()}
                            outputs = text_encoder(**inputs)
                            last_hidden_state = outputs.last_hidden_state[:, 0, :].detach().cpu()
                            last_hidden_states.append(last_hidden_state)
                            del inputs, outputs, last_hidden_state
                            torch.cuda.empty_cache()
                        self.embeddings_text.append(last_hidden_states)
                        # if i > 10:
                        #     break
                torch.cuda.empty_cache()
                self.embeddings_text = torch.stack([torch.stack(inner, dim=0) for inner in self.embeddings_text], dim=0).squeeze(-2).cpu()
                torch.cuda.empty_cache()
                
                if multimodal_type == 'text':
                    self.embeddings = self.embeddings_text
                    torch.cuda.empty_cache()
                    torch.save(self.embeddings, path)       
                    return self.embeddings
                
            self.embeddings = torch.cat((self.embeddings_image, self.embeddings_text), dim=-2)
            torch.cuda.empty_cache()
            torch.save(self.embeddings, path)       
                
        return self.embeddings

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.x[idx]
        image = self.embeddings[idx] if hasattr(self, 'embeddings') else None
        y = self.y[idx]

        return x, image, y

