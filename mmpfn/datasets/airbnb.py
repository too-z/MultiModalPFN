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


class AirbnbDataset(Dataset):
    def __init__(self, data_path):
        
        self.data_path = data_path
        
        FILENAME = 'cleansed_listings_dec18.csv'
        categorical_var = ['host_location', 'host_since_year','host_is_superhost', 'host_neighborhood', 'host_has_profile_pic', 'host_identity_verified', 'neighborhood', 'city', 'smart_location', 'suburb', 'state', 'is_location_exact', 'property_type', 'room_type', 'bed_type', 'instant_bookable', 'cancellation_policy', 'require_guest_profile_picture', 'require_guest_phone_verification', 'host_response_time', 'calendar_updated', 'host_verifications', 'last_review_year']
        numerical_var = ['host_response_rate', 'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'calculated_host_listings_count', 'reviews_per_month']
        text_var = 'description'
        N_CLASSES = 10 # number of classes in classification task
        
        df = pd.read_csv(os.path.join(data_path, FILENAME))
        
        # target binning
        bin_edges = np.quantile(df['price'], q = np.arange(N_CLASSES+1)/10)
        bin_edges[0] = 0 # start at 0
        labels = np.arange(N_CLASSES)
        df['Y'] = pd.cut(df['price'], bins = bin_edges, labels = labels)
        
        # concatenate text fields
        df = df[~((df.summary.isnull()) & (df.description.isnull()))] # drop rows where both fields are empty
        df.loc[df.name.isnull(),'name']= '' # replace NaN name with ''
        df.loc[df.summary.isnull(),'summary']= '' # replace NaN summary with ''
        df.loc[df.description.isnull(),'description']= '' # replace NaN description with ''
        df[text_var] = df['name'] + ' ' + df['summary']+ ' ' + df['description']
        df['host_since_year'] = df['host_since'].str.extract('.*(\d{4})', expand = False) # feature extraction
        df['last_review_year'] = df['last_review'].str.extract('.*(\d{4})', expand = False)
        df['host_response_rate'] = df['host_response_rate'].str.replace('%','')
        df = df[categorical_var + numerical_var + [text_var, 'Y']] # drop unused columns
        df = df.dropna().reset_index(drop=True) # drop na
        df['host_response_rate'] = df['host_response_rate'].astype(int) # format
        
        self.y = df['Y'].values
        self.text = df[[text_var]]
        df = df.drop(columns=['Y', text_var])

        ordianl_encoder = OrdinalEncoder()
        self.x = ordianl_encoder.fit_transform(df[categorical_var])
        self.x = pd.concat([pd.DataFrame(self.x, columns=categorical_var), df[numerical_var]], axis=1).values
        
        
        
    def get_embeddings(self, save=True):
        
        path = f'embeddings/airbnb/airbnb.pt'

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
                text_encoder = AutoModel.from_pretrained(local_dir, local_files_only=True).cuda().eval()
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
                text_encoder = AutoModel.from_pretrained(model_name).cuda().eval()

            self.embeddings = []
            with torch.no_grad():
                for _, texts in tqdm(self.text.iterrows()):
                    last_hidden_states = []
                    for text in texts:
                        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                        inputs = {key: value.to('cuda') for key, value in inputs.items()}
                        outputs = text_encoder(**inputs)
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
