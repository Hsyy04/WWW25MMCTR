set -x

if [ ! -d "dataset" ]; then
    mkdir -p dataset
fi

# 是否已经有了gpt_embedding.npy
if [ ! -f "dataset/gpt_embedding.npy" ]; then
    python -u get_gpt_embedding.py
    if [ ! -f "dataset/gpt_embedding.npy" ]; then
        echo "Error: gpt_embedding.npy was not generated successfully."
        exit 1
    fi
fi
echo "gpt_embedding.npy was generated successfully."

# 是否已经有了item_feature_gpt.parquet
if [ ! -f "dataset/MicroLens_1M_MMCTR/item_feature_gpt.parquet" ]; then
    python -u merge_data.py
    if [ ! -f "dataset/MicroLens_1M_MMCTR/item_feature_gpt.parquet" ]; then
        echo "Error: item_feature_gpt.parquet was not generated successfully."
        exit 1
    fi
fi
echo "item_feature_gpt.parquet was generated successfully."

# 是否已经有了./data/MicroLens_1M_x1/EMB_all_emb_cb05_data.parquet
if [ ! -f "data/MicroLens_1M_x1/EMB_all_emb_cb05_data.parquet" ]; then
    python run_param_tuner.py --config config/EMB_all_emb_cb05.yaml --gpu 0 --script run_all_embedding
    if [ $? -ne 0 ]; then
        echo "train item embedding failed to execute."
        exit 1
    fi
    python encode_all_emb.py --config config/EMB_all_emb_cb05 --expid EMB_cb_allemb_001_89dd7fc0 --gpu 0
    if [ $? -ne 0 ]; then
        echo "encode item embedding failed to execute."
        exit 1
    fi
    python generate_item_info.py --data_name EMB_all_emb_cb05_data
    if [ ! -f "data/MicroLens_1M_x1/EMB_all_emb_cb05_data.parquet" ]; then
        echo "Error: EMB_all_emb_cb05_data.parquet was not generated successfully."
        exit 1
    fi
fi

# 训练DIN
python run_param_tuner.py --config config/DIN_microlens_mmctr_tuner_config_qvq.yaml --gpu 0
if [ $? -ne 0 ]; then
    echo "train DIN failed to execute."
    exit 1
fi
# 生成结果
python prediction.py --config config/DIN_microlens_mmctr_tuner_config_qvq --expid DIN_MicroLens_1M_x1_001_22cde3b8 --gpu 0