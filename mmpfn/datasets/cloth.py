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


class ClothDataset(Dataset):
    def __init__(self, data_path):
        
        self.data_path = data_path
        
        FILENAME = 'Womens Clothing E-Commerce Reviews.csv'
        categorical_var = ['Division Name', 'Department Name', 'Class Name']
        numerical_var = ['Age', 'Positive Feedback Count']
        text_var = 'Review'
            
        df = pd.read_csv(os.path.join(data_path, FILENAME))
        
        df["rating"] = df["Rating"].copy() # rename label
        df = df.rename({"Rating":"Y"}, axis=1) 
        df['Y'] = df['Y'] - 1 # starts from 0
        df.loc[df.Title.isnull(),'Title'] = '' # replace NaN title with ''
        df.loc[df['Review Text'].isnull(),'Title'] = '' # drop NaN reviews (as title is too short)
        df[text_var] = df['Title'] + ' ' + df['Review Text'] # concatenate title and review text
        df = df.dropna().reset_index() # drop na
        df = df[categorical_var + numerical_var + [text_var, 'Y']] # drop unused columns
        
        self.y = df['Y'].values
        self.text = df[[text_var]]
        df = df.drop(columns=['Y', text_var])

        ordianl_encoder = OrdinalEncoder()
        self.x = ordianl_encoder.fit_transform(df[categorical_var])
        self.x = pd.concat([pd.DataFrame(self.x, columns=categorical_var), df[numerical_var]], axis=1).values
        
        
        
    def get_embeddings(self, save=True):
        
        path = f'embeddings/cloth/cloth.pt'

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
