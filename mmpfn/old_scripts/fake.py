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


class FakeDataset(Dataset):
    def __init__(self, data_path):
        
        self.data_path = data_path
        
        FILENAME = 'fake_job_postings.csv'
        categorical_val = ['location', 'salary_range', 'department', 'telecomuting', 'has_company_logo', 'has_question', 'employment_type', 'required_experience', 'rquired_education', 'industry', 'function']
        text_var = ['title', 'company_profile', 'description', 'requirements', 'benefits']

        df = pd.read_csv(os.path.join(data_path, FILENAME))

        df = df.rename({"fraudulent":"Y"}, axis=1)
        df = df[categorical_val + text_var + ['Y']]
        df[categorical_val] = df[categorical_val].astype(str)

        le = LabelEncoder()
        df['Y'] = le.fit_transformer(df['Y'])

        self.y = df['Y'].values
        self.text = df[text_var]
        self.text.fillna('', inplace=True)

        df = df.drop(columns=text_var + ['Y'])

        ordinal_encoder = OrdinalEncoder()
        self.x = ordinal_encoder.fit_transform(df[categorical_val])
        
    
    def get_embeddings(self, save=True):
        
        path = f'embeddings/fake/fake.pt'

        if os.path.exists(path):
            print(f"Load embeddings from {path}")
            self.embeddings = torch.load(path)
        else:
            # Load pretrained DeBERTa-v3 (base version here, can also use small/large)
            # model_name = "microsoft/deberta-v3-base"
            # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            model_name = "google/electra-base-discriminator"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).cuda().eval()

            self.embeddings = []
            with torch.no_grad():
                for i, texts in tqdm(self.text.iterrows()):
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
            print(f"Embeddings shape: {self.embeddings.shape}")
            torch.save(self.embeddings, path)
        
        return self.embeddings
            

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        image = self.embeddings[idx//self.batch_size] if hasattr(self, 'embeddings') else None
        y = self.y[idx]

        return x, image, y
