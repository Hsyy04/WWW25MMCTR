
## 依赖

GPU： Tesla T4；显存16G

+ torch==1.13.1+cu117
+ fuxictr==2.3.7
+ openai

```sh
conda create -n fuxictr python==3.9
pip install -r requirements.txt
source activate fuxictr
conda install openai
```


## 思路概述
观察到DIN学习连续特征没有离散特征快, 我们将item的高质量连续特征进行离散化. 连续多模态特征选择
    1. OpenAI提供的GPT对`item_title`字段的1024维embedding, 
    2. 比赛方提供的img_emb_CLIPRN50的编码.

我们首先对上述两个特征进行离散化, 使用VQ-VAE, 两部分特征分别生成512维度的离散向量, 向量值得范围为[0..1023], 拼接作为新的特征dis_dim, 之后将其作为一个type为sequence的特征传入DIN. 最后, 对于每个item, 我们查找了item_tags\likes_level\views_level\dis_dim四个字段的特征用于DIN的序列建模, 其余保持不变.

##  数据路径

原始数据：
```txt
    ./data/
        MicroLens_1M_x1/
            item_info.parquet
            test.parquet
            train.parquet
            valid.parquet

    ./dataset/
        MicroLens_1M_MMCTR/
            item_feature.parquet
            item_emb.parquet   
            item_seq.parquet  
            item_images.rar  
```

第1步之后：
```txt
    ./data/
        MicroLens_1M_x1/
            item_info.parquet
            test.parquet
            train.parquet
            valid.parquet

    ./dataset/
        MicroLens_1M_MMCTR/
            item_feature.parquet
            ...
        gpt_embedding.npy
```

第2步之后：
```txt
    ./data/
        MicroLens_1M_x1/
            item_info.parquet
            test.parquet
            train.parquet
            valid.parquet

    ./dataset/
        MicroLens_1M_MMCTR/
            item_feature.parquet
            item_feature_gpt.parquet
            ...
        gpt_embedding.npy
```

第3步之后：
```txt
    ./data/
        MicroLens_1M_x1/
            item_info.parquet
            test.parquet
            train.parquet
            valid.parquet
            EMB_all_emb_cb05_data.parquet

    ./dataset/
        MicroLens_1M_MMCTR/
            item_feature.parquet
            item_feature_gpt.parquet
            ...
        gpt_embedding.npy
```

我们处理好数据放在谷歌网盘里：[Download](https://drive.google.com/drive/folders/1gBLHc1lXqW1IqyihuiBVuExz2LX3sJ_n?usp=sharing)

比赛数据下载连接： https://recsys.westlake.edu.cn/MicroLens_1M_MMCTR

## 运行步骤

1. 确保已经存在`dataset/MicroLens_1M_MMCTR/item_feature.parquet`, 使用GPTAPI获取数据中`item_title`字段的embedding, 得到维度为 $\mathcal{R}^1024$ 的特征向量, "dataset/embedding.npy"

```sh
    python -u get_gpt_embedding.py
```

2. 确保已经存在了`dataset/gpt_embedding.npy`和`dataset/MicroLens_1M_MMCTR/item_feature.parquet`. 将向量拼在原始的parquet的文件中，得到（dataset/MicroLens_1M_MMCTR/item_feature_gpt.parquet）

```sh
    python -u merge_data.py
```

3.  确保已经存在了`dataset/MicroLens_1M_MMCTR/item_feature_gpt.parquet`, 之后首先训练一个vae, 之后将特征离散化.

```sh
    python run_param_tuner.py --config config/EMB_all_emb_cb05.yaml --gpu 0 --script run_all_embedding # 训练
    python encode_all_emb.py --config config/EMB_all_emb_cb05 --expid EMB_cb_allemb_001_89dd7fc0 --gpu 0 # 标注
    python generate_item_info.py --data_name EMB_all_emb_cb05_data # 生成item_info文件
```

4.  确保已经存在了`data/MicroLens_1M_x1/EMB_all_emb_cb05_data.parquet`, 之后训练DIN并预测结果

```sh
    python run_param_tuner.py --config config/DIN_microlens_mmctr_tuner_config_qvq.yaml --gpu 0
    python prediction.py --config config/DIN_microlens_mmctr_tuner_config_qvq --expid DIN_MicroLens_1M_x1_001_22cde3b8 --gpu 0
```


### 一键运行版

`bash -x run.sh`
