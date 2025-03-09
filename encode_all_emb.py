import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import logging
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src as model_zoo
from src.mmctr_dataloader import MMCTRDataLoader
import gc
from tqdm import tqdm
import argparse
import os
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import shutil
from src.codebook import VQVAE

class myProcessor(FeatureProcessor):
    def __init__(self, feature_cols=..., label_col=..., dataset_id=None, data_root="../data/", **kwargs):
        super().__init__(feature_cols, label_col, dataset_id, data_root, **kwargs)
    
    def fit_embedding_col(self, col):
        name = col["name"]
        feature_type = col["type"]
        feature_source = col.get("source", "")
        self.feature_map.features[name] = {"source": feature_source,
                                           "type": feature_type}
        if "feature_encoder" in col:
            self.feature_map.features[name]["feature_encoder"] = col["feature_encoder"]
        if "vocab_size" in col:
            self.feature_map.features[name]["vocab_size"] = col["vocab_size"]
        if "embedding_dim" in col:
            self.feature_map.features[name]["embedding_dim"] = col["embedding_dim"]
        if "embedding_num" in col:
            self.feature_map.features[name]["embedding_num"] = col["embedding_num"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())
    
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    feature_encoder = myProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = \
        build_dataset(feature_encoder, **params)
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    all_ddf = pd.read_parquet(params["item_info"])
    pre_ddf = pd.read_parquet("data/MicroLens_1M_x1/item_info.parquet")
    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)
    print(model.checkpoint)

    model.eval()
    # new_emb = []
    dis_emb= []
    for index, batch_data in tqdm(all_ddf.iterrows(), total=len(all_ddf)):
        inputs = dict()
        for col in feature_map.features.keys():
            if col=="item_tags":
                inputs[col] = np.array(batch_data[col]).tolist()
                while len(inputs[col]) < params['tags_pad_len']:
                    inputs[col].append(0)
                inputs[col] = np.array(inputs[col][:params['tags_pad_len']])
                inputs[col] = torch.tensor(inputs[col]).unsqueeze(0).to(model.device)
            else:
                inputs[col] = torch.tensor(np.array(batch_data[col])).unsqueeze(0).to(model.device)

        dis_embedding = model.encode(inputs)
        # new_emb.append(new_embedding.squeeze().cpu().tolist())
        dis_emb.append(dis_embedding.squeeze().cpu().tolist())
    
    # all_ddf['new_emb'] = new_emb
    all_ddf['dis_emb'] = dis_emb
    new_ddf = all_ddf[["item_id", "dis_emb"]]
    new_ddf = pd.merge(pre_ddf, new_ddf, on="item_id", how="left")
    new_ddf.at[0, "dis_emb"]=[0 for i in range(len(dis_emb[0]))]

    save_name = os.path.join("data/MicroLens_1M_x1/", Path(args['config']).name+'.parquet')
    new_ddf.to_parquet(save_name, index=False)

