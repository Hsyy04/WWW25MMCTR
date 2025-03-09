from fuxictr.pytorch.layers import FeatureEmbedding, FeatureEmbeddingDict
from torch import nn
import torch
import copy

class DiscEmbedding(FeatureEmbedding):
    def __init__(self, feature_map, embedding_dim, embedding_initializer="partial(nn.init.normal_, std=1e-4)", required_feature_columns=None, not_required_feature_columns=None, use_pretrain=True, use_sharing=True, bucket_num = 20, dropout_rates=0.5):
        super().__init__(feature_map, embedding_dim, embedding_initializer, required_feature_columns, not_required_feature_columns, use_pretrain, use_sharing)
        
        self.embedding_names = []
        for feature, feature_spece in feature_map.features.items():
            if feature_spece['type'] == "embedding":
                self.embedding_names.append(feature)
        self.embedding_layer = FeatureEmbeddingDict(feature_map, 
                                            embedding_dim,
                                            embedding_initializer=embedding_initializer,
                                            required_feature_columns=required_feature_columns,
                                            not_required_feature_columns=not_required_feature_columns,
                                            use_pretrain=use_pretrain,
                                            use_sharing=use_sharing)
        self.disc_layer = nn.ModuleDict()
        self.codebook_embedding = nn.ModuleDict()
        self.trans = nn.ModuleDict()
        self.bucket_num = bucket_num
        for feature in self.embedding_names:
            feature_spece = feature_map.features[feature]
            feat_dim = feature_spece.get("embedding_dim", embedding_dim)
            self.disc_layer[feature] = nn.BatchNorm1d(feat_dim)
            self.codebook_embedding[feature] = nn.Linear(feat_dim*(self.bucket_num+1), feat_dim)
            self.trans[feature] = nn.Linear(feat_dim, feat_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rates)
        self.mse = nn.MSELoss()
        self.feature_map = feature_map

    def forward(self, X, feature_source=[], feature_type=[], flatten_emb=False):
        # 在encoder之前重新处理embedding类型得特征
        # 添加一个重建损失学习离散化矩阵
        res_loss = None
        for feature, values in X.items():
            if feature in self.embedding_names:
                emb = self.disc_layer[feature](values.float()) # bz * emb_dim -> bz * emb_dim normlize
                emb = torch.clamp(emb, min=-3, max=3)
                emb = (emb+3)/6 * self.bucket_num
                emb = torch.floor(emb).long() # bz * emb_dim
                emb = torch.nn.functional.one_hot(emb, num_classes=self.bucket_num+1).float()# bz * emb_dim * bucket_num
                emb = emb.view(emb.shape[0], -1) # bz * (emb_dim * bucket_num)
                emb = self.codebook_embedding[feature](emb) # bz * (emb_dim)
                final_embedding = self.activation(emb)
                if self.training:
                    res_embedding = self.trans[feature](final_embedding) # decoder bz * emb_dim
                    res_loss = self.mse(res_embedding, values.float())
                X[feature] = final_embedding
        feature_emb_dict = self.embedding_layer(X, feature_source=feature_source, feature_type=feature_type)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=flatten_emb)
        if self.training and res_loss is not None:
            return feature_emb, res_loss
        else:
            return feature_emb, None
        

class SeqdiscEmbedding(FeatureEmbedding):
    def __init__(self, feature_map, embedding_dim, embedding_initializer="partial(nn.init.normal_, std=1e-4)", required_feature_columns=None, not_required_feature_columns=None, use_pretrain=True, use_sharing=True, bucket_num = 20, dropout_rates=0.5):
        super().__init__(feature_map, embedding_dim, embedding_initializer, required_feature_columns, not_required_feature_columns, use_pretrain, use_sharing)
        self.embedding_names = ["dis_emb"]
        self.feature_spec = feature_map.features["dis_emb"]
        self.embedding_layer = FeatureEmbeddingDict(feature_map, 
                                            embedding_dim,
                                            embedding_initializer=embedding_initializer,
                                            required_feature_columns=required_feature_columns,
                                            not_required_feature_columns=not_required_feature_columns,
                                            use_pretrain=use_pretrain,
                                            use_sharing=use_sharing)
        self.codebook_embedding = nn.Embedding(self.feature_spec['vocab_size'], self.feature_spec['embedding_dim'])
        self.trans = nn.Linear(self.feature_spec['embedding_dim']*self.feature_spec['embedding_num'], self.feature_spec['embedding_dim'])
        self.activation = nn.ReLU()
        self.feature_map = feature_map

    def forward(self, X, feature_source=[], feature_type=[], flatten_emb=False):
        for feature in X:
            if feature not in self.embedding_names:
                continue
            values = X[feature]
            emb = self.codebook_embedding(values.long())
            emb = self.trans(emb.view(values.shape[0], -1))
            final_embedding = self.activation(emb)
            X[feature] = final_embedding
        feature_emb_dict = self.embedding_layer(X, feature_source=feature_source, feature_type=feature_type)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=flatten_emb)
        return feature_emb
    
class AddEmbedding(FeatureEmbedding):
    def __init__(self, feature_map, embedding_dim, add_embedding_dim, embedding_initializer="partial(nn.init.normal_, std=1e-4)", required_feature_columns=None, not_required_feature_columns=None, use_pretrain=True, use_sharing=True, bucket_num = 20, dropout_rates=0.5):
        super().__init__(feature_map, embedding_dim, embedding_initializer, required_feature_columns, not_required_feature_columns, use_pretrain, use_sharing)
        self.embedding_names = []
        for feature, feature_spece in feature_map.features.items():
            if feature_spece['type'] == "embedding":
                self.embedding_names.append(feature)
        self.embedding_layer = FeatureEmbeddingDict(feature_map, 
                                            embedding_dim,
                                            embedding_initializer=embedding_initializer,
                                            required_feature_columns=required_feature_columns,
                                            not_required_feature_columns=not_required_feature_columns,
                                            use_pretrain=use_pretrain,
                                            use_sharing=use_sharing)
        print(self.embedding_layer.feature_encoders.keys())
        self.disc_layer = nn.ModuleDict()
        self.add_embedding = nn.ModuleDict()
        self.trans = nn.ModuleDict()
        self.bucket_num = bucket_num
        for feature in self.embedding_names:
            feature_spece = feature_map.features[feature]
            feat_dim = feature_spece.get("embedding_dim", embedding_dim)
            self.add_embedding[feature] = nn.Linear(feat_dim, feat_dim)
            self.trans[feature] = nn.Linear(feat_dim, feat_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rates)
        self.feature_map = feature_map

    def forward(self, X, feature_source=[], feature_type=[], flatten_emb=False):
        for feature in X:
            if feature not in self.embedding_names:
                continue
            values = X[feature]
            emb = self.add_embedding[feature](values.float())
            emb = self.activation(emb)
            emb = self.dropout(emb)
            final_embedding = self.trans[feature](emb)
            X[feature] = final_embedding
        feature_emb_dict = self.embedding_layer(X, feature_source=feature_source, feature_type=feature_type)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=flatten_emb)
        return feature_emb