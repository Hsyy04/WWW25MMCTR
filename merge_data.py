import pandas as pd
import numpy as np
import time
print("loading data")
raw_data = pd.read_parquet("dataset/MicroLens_1M_MMCTR/item_feature.parquet")
gpt_data = np.load("dataset/gpt_embedding.npy").tolist()
print("data loaded")
raw_data["txt_emb_GPT"] = gpt_data
print("saving data")
raw_data.to_parquet("dataset/MicroLens_1M_MMCTR/item_feature_gpt.parquet")
print("saving done")
time.sleep(1)