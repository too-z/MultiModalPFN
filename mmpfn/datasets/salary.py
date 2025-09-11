import os
import torch
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel

from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from pathlib import Path
from tqdm import tqdm
from mmpfn.models.dino_v2.models.vision_transformer import vit_base


class SalaryDataset(Dataset):
    def __init__(self, data_path):
        
        self.data_path = data_path
        
        FILENAME = 'train.csv'
        categorical_var = ['location', 'company_name_encoded']
        numerical_var = ['experience_int']
        text_var = 'description'
        
        df = pd.read_csv(os.path.join(data_path, FILENAME))
        
        df = df.rename({"salary":"Y"}, axis=1) # rename label
        # compute years of experience
        df['experience_int'] = df['experience'].str.split("-").str.get(0)
        # concatenate text fields
        df.loc[df.job_description.isnull(),'job_description']= '' # replace NaN job_description with ''
        df.loc[df.job_desig.isnull(),'job_desig']= '' # replace NaN job_desig with ''
        df.loc[df.key_skills.isnull(),'key_skills']= '' # replace NaN key_skills with ''
        df[text_var] = df['job_description'] + ' ' + df['job_desig']+ ' ' + df['key_skills']
        df = df[categorical_var + numerical_var + [text_var, 'Y']] # drop unused columns
        df = df.dropna().reset_index(drop=True) # drop na
        df[categorical_var] = df[categorical_var].astype(str) # format
        df[numerical_var] = df[numerical_var].astype(int) # format 
        
        le = LabelEncoder()
        df['Y'] = le.fit_transform(df['Y']) # label encoding of target variable
        
        self.y = df['Y'].values
        self.text = df[text_var].values
        df = df.drop(columns=['Y', text_var])

        ordianl_encoder = OrdinalEncoder()
        self.x = ordianl_encoder.fit_transform(df[categorical_var])
        self.x = pd.concat([pd.DataFrame(self.x, columns=categorical_var), df[numerical_var]], axis=1).values
        
        
    def get_embeddings(self, save=True):
        
        path = f'embeddings/salary/salary.pt'

        if os.path.exists(path):
            print(f"Load embeddings from {path}")
            self.embeddings = torch.load(path)
        else:
            # Load pretrained DeBERTa-v3 (base version here, can also use small/large)
            model_name = "microsoft/deberta-v3-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            model = AutoModel.from_pretrained(model_name).cuda().eval()

            self.embeddings = []
            with torch.no_grad():
                for text in tqdm(self.text):
                    inputs = tokenizer(text, return_tensors="pt") # Tokenize and convert to tensors
                    inputs = {key: value.to('cuda') for key, value in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs) # Forward pass
                    last_hidden_state = outputs.last_hidden_state # outputs.last_hidden_state shape: [batch_size, seq_len, hidden_dim]
                    self.embeddings.append(last_hidden_state[:, 0, :])  # shape: [batch_size, hidden_dim], CLS token is always at index 0

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
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        image = self.embeddings[idx//self.batch_size] if hasattr(self, 'embeddings') else None
        y = self.y[idx]

        return x, image, y
