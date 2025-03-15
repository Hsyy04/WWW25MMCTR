import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="EMB_all_emb_cb08_d")
args = parser.parse_args()
data_name = args.data_name

new_emb = pd.read_parquet(f"data/MicroLens_1M_x1/{data_name}.parquet")
item_feature = pd.read_parquet("dataset/MicroLens_1M_MMCTR/item_feature.parquet")[["item_id", "likes_level",  "views_level"]]

all_emb = pd.merge(new_emb, item_feature, on="item_id", how="left")
all_emb = all_emb.rename(columns={"likes_level": "likes", "views_level": "views"})
print(all_emb.head())
all_emb.to_parquet(f"data/MicroLens_1M_x1/{data_name}.parquet")