# 把离散化的特征切成k部分

import pandas as pd
import numpy as np
import yaml 
import os
import torch
from tqdm import tqdm

feature_num = 2
base_config_name = "DIN_microlens_mmctr_tuner_config_allemb15.yaml"
base_config_name_new = "DIN_microlens_mmctr_tuner_config_allemb15dis.yaml"

base_config = yaml.load(open(f"config/{base_config_name}", "r"), Loader=yaml.FullLoader)
feature_cols = base_config["dataset_config"]['MicroLens_1M_x1']['feature_cols']
info_path = base_config["dataset_config"]['MicroLens_1M_x1']['item_info']

# 将dis_emb切成feature_num部分
info_data = pd.read_parquet(info_path)
dis_embs = [item.tolist() for item in info_data["dis_emb"]]
dis_embs = torch.LongTensor(dis_embs)
features = torch.split(dis_embs, 1024//feature_num, dim=1)

info_data = info_data.drop(columns=["dis_emb"])
new_feature_cols = []
for item in feature_cols:
    if item['name'] == "dis_emb":
        continue
    new_feature_cols.append(item)
feature_cols = new_feature_cols
# 生成新的config
for i, feature_i in tqdm(enumerate(features), total=feature_num):
    info_data[f"feature_{i}"] = feature_i.tolist()
    feature_cols.append({
        "name": f"feature_{i}",
        "active": True,
        "dtype":"int",
        "type":"sequence",
        "max_len": 1024//feature_num,
        "vocab_size": feature_i.max().item() + 1,
        "source":"item"
    })



base_config["dataset_config"]['MicroLens_1M_x1']['feature_cols'] = feature_cols
save_info_path = f"{info_path[:-8]}_{feature_num}.parquet"
base_config["dataset_config"]['MicroLens_1M_x1']['item_info'] = save_info_path

yaml.dump(base_config, open(f"config/{base_config_name_new}", "w"), default_flow_style=False)
info_data.to_parquet(save_info_path)
print("Done!")


