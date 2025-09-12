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
        self.text = df[[text_var]]
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
            local = False
            # model_name = "microsoft/deberta-v3-base" 
            # local_dir = ".dataset/deberta"
            model_name = "google/electra-base-discriminator"
            local_dir = ".dataset/electra"
            if 'deberta' in model_name:
                use_fast = False
            else:
                use_fast = True
            
            if local:
                tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=use_fast, local_files_only=True)
                model = AutoModel.from_pretrained(local_dir, local_files_only=True).cuda().eval()
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
                model = AutoModel.from_pretrained(model_name).cuda().eval()

            self.embeddings = []
            with torch.no_grad():
                for _, texts in tqdm(self.text.iterrows()):
                    last_hidden_states = []
                    for text in texts:
                        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                        inputs = {key: value.to('cuda') for key, value in inputs.items()}
                        outputs = model(**inputs)
                        last_hidden_state = outputs.last_hidden_state[:, 0, :].detach().cpu()
                        last_hidden_states.append(last_hidden_state)
                        del inputs, outputs, last_hidden_state
                        torch.cuda.empty_cache()
                    self.embeddings.append(last_hidden_states)

            self.embeddings = torch.stack([torch.stack(inner, dim=0) for inner in self.embeddings], dim=0).squeeze(-2)
            torch.save(self.embeddings, path)
        
        return self.embeddings
            

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        image = self.embeddings[idx//self.batch_size] if hasattr(self, 'embeddings') else None
        y = self.y[idx]

        return x, image, y
