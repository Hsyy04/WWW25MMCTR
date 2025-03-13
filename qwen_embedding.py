import dashscope
import base64
import json
import retry
from http import HTTPStatus
from tqdm import tqdm
import numpy as np
import time
import os
dashscope.api_key = "sk-7d189a44c48043ec8fafd3d98e22251e"
# 读取图片并转换为Base64,实际使用中请将xxx.png替换为您的图片文件名或路径

@retry.retry(tries=3, delay=2)
def get_embedding(image_path):
    with open(image_path, "rb") as image_file:
        # 读取文件并转换为Base64
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    # 设置图像格式
    image_format = "png"  # 根据实际情况修改，比如jpg、bmp 等
    image_data = f"data:image/{image_format};base64,{base64_image}"
    # 输入数据
    inputs = [{'image': image_data}]

    # 调用模型接口
    resp = dashscope.MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        input=inputs
    )
    if resp.status_code == HTTPStatus.OK:
       output = resp.output
    else:
        raise Exception(f"Failed to call multimodal-embedding-v1, status code: {resp.status_code}, message: {resp.message}")
    
    ret_embedding = output["embeddings"][0]["embedding"]
    return ret_embedding

dir_path = "dataset/MicroLens_1M_MMCTR/item_images"
all_embedding = np.load("dataset/MicroLens_1M_MMCTR/qw_data.npy").tolist()
start_id = len(all_embedding)+1
print(start_id)
assert False
image_files = os.listdir(dir_path)
image_files = sorted(image_files)


for i in tqdm(range(start_id, 91719), total=91719-len(all_embedding)):
    image_path = os.path.join(dir_path, f"{i}.jpg")
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue
    embedding = get_embedding(image_path)
    all_embedding.append(embedding)
    time.sleep(0.3)

    if i % 10000 == 0:
        qw_data = np.array(all_embedding)
        np.save("dataset/MicroLens_1M_MMCTR/qw_data.npy", qw_data)


import  numpy as np

qw_data = np.array(all_embedding)
np.save("dataset/MicroLens_1M_MMCTR/qw_data.npy", qw_data)