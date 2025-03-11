import torch

import torch.nn as nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.torch_utils import get_device, get_optimizer
from tqdm import tqdm
import numpy as np
import sys
import time
import logging
from fuxictr.pytorch import layers

class VQVAE(BaseModel):
    def __init__(self, 
                 feature_map=None,
                 model_id="VQVAE", 
                 num_embeddings=512, # 词典大小
                 embedding_dim=128, # 向量表里的大小
                 output_dim=1024, # 输出的大小
                 hidden_dim = 8192,  # encoder的隐藏层大小
                 model_alpha=0.5, 
                 model_beta=0.5, 
                 learning_rated=1e-3, 
                 **kwargs):
        super(VQVAE, self).__init__(feature_map, model_id, **kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.output_dim = output_dim
        self.alpha = model_alpha
        self.beta = model_beta
        self.accumulation_steps = kwargs.get("accumulation_steps", 1)

        self.encoder = nn.Sequential( #TODO: 这个用cnn会更好，可能不需要这么多参数
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim*output_dim)
        )

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim*output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )

        self.optimizer = get_optimizer(kwargs["optimizer"], self.parameters(), learning_rated)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, x):
        time.sleep(0.1)
        x = x.float()
        z_e = self.encoder(x) # n*128 -> n*d
        bn = z_e.size(0)
        z_e = z_e.view(bn, self.output_dim, self.embedding_dim) # n*(Ld) -> n*L*d

        distances = (torch.sum(z_e ** 2, dim=2, keepdim=True) 
                        + torch.sum(self.codebook.weight ** 2, dim=1)
                        - 2 * torch.matmul(z_e, self.codebook.weight.t())) # n*L*d->n*L*n_embedding

        encoding_indices = torch.argmin(distances, dim=2)# n*L*n_embedding -> n*L
        z_q = self.codebook(encoding_indices)# n*L->n*L*d

        decoder_input = z_e + (z_q - z_e).detach()
        x_recon = self.decoder(decoder_input.view(bn, -1)) # n*L*d -> n*128
        return {
            "x_recon": x_recon,
            "x": x,
            "z_e": z_e,
            "z_q": z_q,
        }
    
    def train_step(self, batch_data):
        batch_data = self.get_inputs(batch_data)
        return_dict = self.forward(batch_data)
        loss = self.compute_loss(return_dict)
        loss = loss / self.accumulation_steps
        loss.backward()
        if (self._batch_index + 1) % self.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss
    
    def get_inputs(self, inputs):
        if isinstance(inputs, dict):
            batch_data = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            batch_data = inputs.to(self.device)
        return batch_data

    def compute_loss(self, return_dict,):
        x_recon = return_dict["x_recon"]
        x = return_dict["x"]
        z_e = return_dict["z_e"]
        z_q = return_dict["z_q"]
        loss_recon = F.mse_loss(x_recon, x)
        loss_e = F.mse_loss(z_q.detach(), z_e)
        loss_q = F.mse_loss(z_e.detach(), z_q)
        loss = loss_recon + self.alpha*loss_q + self.beta * loss_e
        return loss
    
    def evaluate(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            recon_loss = []
            all_loss = []
            if self._verbose > 0:
                # data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
                data_generator = data_generator
            for batch_data in data_generator:
                batch_data = self.get_inputs(batch_data)
                return_dict = self.forward(batch_data)
                loss = F.mse_loss(return_dict["x_recon"], return_dict["x"])
                recon_loss.append(loss.item())
                all_loss.append(self.compute_loss(return_dict).item())
            recon_loss_mean = np.mean(recon_loss)
            all_loss_mean = np.mean(all_loss)
            val_logs = {}
            val_logs["rec_loss"] = recon_loss_mean
            val_logs["all_loss"] = all_loss_mean
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            return val_logs

    def encode(self, x):
        x = x.float()
        z_e = self.encoder(x) # n*128 -> n*d
        bn = z_e.size(0)
        z_e = z_e.view(bn, self.output_dim, self.embedding_dim) # n*(Ld) -> n*L*d

        distances = (torch.sum(z_e ** 2, dim=2, keepdim=True) 
                        + torch.sum(self.codebook.weight ** 2, dim=1)
                        - 2 * torch.matmul(z_e, self.codebook.weight.t()))

        encoding_indices = torch.argmin(distances, dim=2)# n*L*n_embedding -> n*L
        z_q = self.codebook(encoding_indices)# n*L->n*L*d
        return z_q, encoding_indices
    

class VQVAEAllEmb(VQVAE):
    def __init__(self, feature_map=None, model_id="VQVAEAllEmb", num_embeddings=512, embedding_dim=128, output_dim=1024, hidden_dim=8192, model_alpha=0.5, model_beta=0.5, learning_rate=0.001, **kwargs):
        super().__init__(feature_map, model_id, num_embeddings, embedding_dim, output_dim, hidden_dim, model_alpha, model_beta, learning_rate, **kwargs)

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.output_dim = output_dim
        self.alpha = model_alpha
        self.beta = model_beta
        self.accumulation_steps = kwargs.get("accumulation_steps", 1)
        

        self.encoder = nn.Sequential( 
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim*output_dim)
        )

        self.embedding = nn.ModuleDict()
        self.encoder = nn.ModuleDict()
        self.codebook = nn.ModuleDict()
        self.decoder = nn.ModuleDict()
        for feature, feature_spec in feature_map.features.items():
            if feature in ["txt_emb_BERT", "img_emb_CLIPRN50"]:
                feat_dim = feature_spec.get("embedding_dim", hidden_dim)
                self.embedding[feature] = nn.Identity()
                self.encoder[feature] = nn.Sequential( #TODO: 这个用cnn会更好，可能不需要这么多参数
                        nn.Linear(feat_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(kwargs['net_dropout']),
                        nn.Linear(hidden_dim, embedding_dim*output_dim)
                    )
                self.codebook[feature]  = nn.Embedding(num_embeddings, embedding_dim)
                self.codebook[feature].weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
                self.decoder[feature] = nn.Sequential(
                    nn.Linear(embedding_dim*output_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(kwargs['net_dropout']),
                    nn.Linear(hidden_dim, feat_dim)
                )
            else:
                self.embedding[feature] = nn.Embedding(feature_spec["vocab_size"], hidden_dim)
                self.encoder[feature] = nn.Linear(hidden_dim, embedding_dim)

                self.codebook[feature] = nn.Embedding(num_embeddings, embedding_dim)
                self.codebook[feature].weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

                self.decoder[feature] = nn.Linear(embedding_dim, feature_spec["vocab_size"])

        self.optimizer = get_optimizer(kwargs["optimizer"], self.parameters(), learning_rate)
        self.reset_parameters()
        self.model_to_device()
    def forward(self, x):
        return_dict = {}
        for feature in x:
            if feature in ["txt_emb_BERT", "img_emb_CLIPRN50"]:
                values = x[feature].float()
                z = self.embedding[feature](values)
                z_e = self.encoder[feature](z) 
                bn = z_e.size(0)
                z_e = z_e.view(bn, self.output_dim, self.embedding_dim)
                distances = (torch.sum(z_e ** 2, dim=2, keepdim=True) 
                                + torch.sum(self.codebook[feature].weight ** 2, dim=1)
                                - 2 * torch.matmul(z_e, self.codebook[feature].weight.t()))
                encoding_indices = torch.argmin(distances, dim=2)
                z_q = self.codebook[feature](encoding_indices)
                decoder_input = z_e + (z_q - z_e).detach()
                x_recon = self.decoder[feature](decoder_input.view(bn, -1))
                return_dict[feature]  = {
                    "x_recon": x_recon,
                    "x": values,
                    "z_e": z_e,
                    "z_q": z_q,
                }
            else:
                values = x[feature].long()
                if values.dim() != 2:
                    values = values.unsqueeze(1)
                z = self.embedding[feature](values)
                z_e = self.encoder[feature](z)
                distances = (torch.sum(z_e ** 2, dim=2, keepdim=True) 
                                + torch.sum(self.codebook[feature].weight ** 2, dim=1)
                                - 2 * torch.matmul(z_e, self.codebook[feature].weight.t()))
                encoding_indices = torch.argmin(distances, dim=2)
                z_q = self.codebook[feature](encoding_indices)
                decoder_input = z_e + (z_q - z_e).detach()
                x_recon = self.decoder[feature](decoder_input)
                return_dict[feature]  = {
                    "x_recon": x_recon.view(-1, x_recon.size(-1)),
                    "x": values.view(-1),
                    "z_e": z_e,
                    "z_q": z_q,
                }

        return return_dict

    def compute_loss(self, return_dict):
        loss = 0
        for feature in return_dict:
            if feature in ["txt_emb_BERT", "img_emb_CLIPRN50"]:
                loss += F.mse_loss(return_dict[feature]["x_recon"], return_dict[feature]["x"])
                loss_e = F.mse_loss(return_dict[feature]["z_q"].detach(), return_dict[feature]["z_e"])
                loss_q = F.mse_loss(return_dict[feature]["z_e"].detach(), return_dict[feature]["z_q"])
                loss += self.alpha*loss_q + self.beta * loss_e
            else:
                loss += F.cross_entropy(return_dict[feature]["x_recon"], return_dict[feature]["x"])
                loss_e = F.mse_loss(return_dict[feature]["z_q"].detach(), return_dict[feature]["z_e"])
                loss_q = F.mse_loss(return_dict[feature]["z_e"].detach(), return_dict[feature]["z_q"])
                loss += self.alpha*loss_q + self.beta * loss_e
        return loss     

    def evaluate(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            recon_loss = []
            all_loss = []
            if self._verbose > 0:
                # data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
                data_generator = data_generator
            for batch_data in data_generator:
                batch_data = self.get_inputs(batch_data)
                return_dict = self.forward(batch_data)
                loss = 0.0
                for feature in return_dict:
                    if feature in ["txt_emb_BERT", "img_emb_CLIPRN50"]:
                        loss += F.mse_loss(return_dict[feature]["x_recon"], return_dict[feature]["x"])
                    else:
                        loss += F.cross_entropy(return_dict[feature]["x_recon"], return_dict[feature]["x"])
                
                recon_loss.append(loss.item())
                all_loss.append(self.compute_loss(return_dict).item())
            recon_loss_mean = np.mean(recon_loss)
            all_loss_mean = np.mean(all_loss)
            val_logs = {}
            val_logs["rec_loss"] = recon_loss_mean
            val_logs["all_loss"] = all_loss_mean
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            return val_logs
    
    def encode(self, x):
        return_dis_emb = []
        for feature in x:
            if feature in ["txt_emb_BERT", "img_emb_CLIPRN50"]:
                values = x[feature].float()
                z = self.embedding[feature](values)
                z_e = self.encoder[feature](z) 
                bn = z_e.size(0)
                z_e = z_e.view(bn, self.output_dim, self.embedding_dim)
                distances = (torch.sum(z_e ** 2, dim=2, keepdim=True) 
                                + torch.sum(self.codebook[feature].weight ** 2, dim=1)
                                - 2 * torch.matmul(z_e, self.codebook[feature].weight.t()))
                encoding_indices = torch.argmin(distances, dim=2)
                return_dis_emb.append(encoding_indices)
            else:
                values = x[feature].long()
                if values.dim() != 2:
                    values = values.unsqueeze(1)
                z = self.embedding[feature](values)
                z_e = self.encoder[feature](z)
                distances = (torch.sum(z_e ** 2, dim=2, keepdim=True) 
                                + torch.sum(self.codebook[feature].weight ** 2, dim=1)
                                - 2 * torch.matmul(z_e, self.codebook[feature].weight.t()))
                encoding_indices = torch.argmin(distances, dim=2)
                return_dis_emb.append(encoding_indices)

        return torch.cat(return_dis_emb, dim=1)