# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


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
from fuxictr.preprocess import FeatureProcessor, build_dataset, split_train_test
import src as model_zoo
from src.mmctr_dataloader import CBAllDataLoader
import gc
import argparse
import os
from pathlib import Path
import pandas as pd

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
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
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

    # split train, valid
    train_ddf = pd.read_parquet(params['item_info'])  # all_info_data
    train_ddf, valid_ddf, test_ddf = split_train_test(train_ddf=train_ddf, valid_size=0.1, split_type="random")
    train_ddf.to_parquet(os.path.join(data_dir, "train.parquet"), index=False)
    valid_ddf.to_parquet(os.path.join(data_dir, "valid.parquet"), index=False)
    params['train_data'] = os.path.join(data_dir, "train.parquet")
    params['valid_data'] = os.path.join(data_dir, "valid.parquet")
    params['test_data'] = None
    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model
    # get load
    train_gen, valid_gen = CBAllDataLoader(feature_map=feature_map, data_path=params['train_data'], **params), CBAllDataLoader(feature_map=feature_map, data_path=params['valid_data'], **params)
    model.fit(train_gen, validation_data=valid_gen, **params)

    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(valid_gen)
    
    result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {}\n' \
            .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                    ' '.join(sys.argv), experiment_id, params['dataset_id'],
                    "N.A.", print_to_list(valid_result)))
