import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm


# Example usage
if __name__ == "__main__":
    all_data = pd.read_parquet("data/MicroLens_1M_x1/EMB_all_emb_cb05_data.parquet")
    dis_embs = np.array(all_data["dis_emb"].tolist()).transpose(1, 0).tolist()
    possible = []
    for data in tqdm(dis_embs, total=len(dis_embs)):
        pinlv = Counter(data)
        for key, value in pinlv.items():
            value = value/len(data)

        possible_one = []
        for i in range(1024):
            if i in pinlv:
                possible_one.append(pinlv[i])
            else:
                possible_one.append(1e-6)
        possible.append(possible_one)

    shangs = []
    for pos in possible:
        shangs.append(-np.sum(pos*np.log2(pos)))

    shangs = np.array(shangs)
    # 找前128个最大的
    index = np.argsort(shangs)[::-1][:128]
    
    new_dis_embs = []
    for i in index:
        new_dis_embs.append(dis_embs[i])
    
    all_data["dis_emb"] = np.array(new_dis_embs).transpose(1, 0).tolist()
    all_data.to_parquet("data/MicroLens_1M_x1/EMB_all_emb_cb05_data_dis128.parquet")
