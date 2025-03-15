import os
import pandas as pd
from openai import OpenAI
import numpy as np
from tqdm import tqdm

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)


def get_embeddings_batch(texts, dim=64, model="text-embedding-3-small"):
    texts = [text.replace("\n", " ") for text in texts]
    response = client.embeddings.create(input=texts, model=model)
    return [normalize_l2(response.data[i].embedding[:dim]) for i in range(len(response.data))]


class Args:
    model, dim, batch_size = 'text-embedding-3-large', 1024, 128
    output_file = 'dataset/gpt_embedding.npy'


df = pd.read_parquet('dataset/MicroLens_1M_MMCTR/item_feature.parquet')
all_title = df['item_title'].values

client = OpenAI(api_key='YOUR_API_KEY')
embeddings = []
for i in tqdm(range(0, len(all_title), Args.batch_size)):
    batch_texts = all_title[i: i + Args.batch_size].tolist()
    batch_embeddings = get_embeddings_batch(batch_texts, dim=Args.dim, model=Args.model)
    embeddings.extend(batch_embeddings)

embeddings = np.array(embeddings)
np.save(Args.output_file, embeddings)