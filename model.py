import pickle

import numpy as np
import torch
import torch.nn as nn
from psbody.mesh import Mesh
from torch.nn import Module

from config import cfg
from modules.hubert import Hubert2Vec
from modules.wave2vec import Wav2Vec2Model
from utils import length_same


# Temporal Bias, brrowed from https://github.com/EvelynFan/FaceFormer/blob/main/faceformer.py

class MamlTalk(Module):

    def __init__(self):
        super(MamlTalk, self).__init__()
        self.dataset = cfg.dataset
        self.personal_dim = cfg.model.personal_dim
        self.backbone = cfg.backbone
        self.upper_map, self.mouth_map = self.get_upper_msk()

        if not cfg.use_pregenerated_feature:
            if self.backbone.lower() == 'hubert':
                self.audio_encoder = Hubert2Vec.from_pretrained(cfg.path.hubert)
            elif self.backbone.lower() == 'wav2vec':
                self.audio_encoder = Wav2Vec2Model.from_pretrained(cfg.path.wav2vec)
            else:
                raise NotImplementedError('audio_backbone {} not implemented'.format(self.backbone))
        else:
            self.audio_encoder = None
        self.template_mesh = Mesh(filename=cfg.path.render_template)
        if cfg.model.freeze_audio and not cfg.use_pregenerated_feature:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        if cfg.dataset == "vocaset":
            with open(cfg.path.pkl, 'rb') as f:
                self.lip_mask = pickle.load(f, encoding='latin1')["lips"]
                self.lip_map = nn.Linear(254 * 3, 1024)

        elif cfg.dataset == "BIWI":
            with open('./BIWI/BIWI_lip.pkl', 'rb') as f:
                self.lip_mask = pickle.load(f, encoding='latin1')
                self.lip_map = nn.Linear(4996 * 3, 1024)
        else:
            raise NotImplementedError
        self.audio_fps = cfg.audio_fps
        self.video_fps = cfg.video_fps

        self.mse_func = nn.MSELoss(reduction='none')
        # if cfg.cross_train:

        self.lip_vertice_mapper = []

        self.motion_layer = nn.Linear(5023 * 3, cfg.feature_dim)
        # decoder_layer = nn.TransformerDecoderLayer(d_model=cfg.feature_dim, nhead=4,
        #                                            dim_feedforward=2 * cfg.feature_dim, batch_first=True)
        self.refer_motion_encoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=cfg.feature_dim, nhead=4,
            dim_feedforward=2 * cfg.feature_dim,
            batch_first=True), num_layers=1
        )

        decoder_layer = nn.TransformerDecoderLayer(d_model=cfg.feature_dim, nhead=4,
                                                   dim_feedforward=2 * cfg.feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        if cfg.dataset == "vocaset":
            self.audio_feature_map = nn.Linear(1024, cfg.feature_dim)
        elif cfg.dataset == "BIWI":
            self.audio_feature_map = nn.Linear(2048, cfg.feature_dim)
        self.vertice_map_r = nn.Linear(cfg.feature_dim, cfg.vertice_dim)

        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def get_upper_msk(self):
        pkl_path = cfg.path.pkl
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        lips = 'lips'
        mouth_map = data[lips]
        eye = 'eye_region'
        eye = list(data[eye])
        forehead = 'forehead'
        forehead = list(data[forehead])

        upperface = eye + forehead
        upper_map = np.unique(upperface)
        return upper_map, mouth_map

    def forward(
            self, audio,vertices_mask
    ):
        # 扩展模型基础为1*batchsize方便直接使用
        template = template.unsqueeze(1)

        if not cfg.use_pregenerated_feature:
            hidden_states = self.audio_encoder(audio).last_hidden_state
        else:
            hidden_states = audio  # bzs,seq,1024
        hidden_states, vertices_mask = length_same(hidden_states, vertices_mask)
        vertice_input = self.audio_feature_map(hidden_states)

        vertice_out = self.transformer_decoder(vertice_input, vertice_input, tgt_key_padding_mask=vertices_mask,
                                               memory_key_padding_mask=vertices_mask)
        vertice_out = self.vertice_map_r(vertice_out)

        vertice_out = vertice_out
        return vertice_out

    def loss_function(
            self, vertices_gt, vertices_pred, vertice_mask,  process='train',
    ):

        vertices_pred, vertices_gt = length_same(vertices_pred, vertices_gt)
        vertice_mask, vertices_gt = length_same(vertice_mask, vertices_gt)
        change_location_mask = torch.abs(change_location[:, 1:] - change_location[:, :-1])

        assert vertices_gt.size() == vertices_pred.size()

        basic_loss = self.mse_func(vertices_gt, vertices_pred) * cfg.loss.l2
        basic_loss1 = basic_loss * vertice_mask.unsqueeze(2)
        basic_loss = torch.sum(basic_loss1) / torch.sum(vertice_mask) / basic_loss.shape[2]
        self.log(f"{process}_l2_loss", basic_loss, prog_bar=True, batch_size=vertices_gt.shape[0])

        if cfg.loss.freq > 0:
            fluency_loss = self.mse_func(
                vertices_gt[:, 1:, :] - vertices_gt[:, :-1, :], vertices_pred[:, 1:] - vertices_pred[:, :-1]
            )
            assert (vertices_gt[:, 1:, :] - vertices_gt[:, :-1, :]).size() == (
                    vertices_pred[:, 1:] - vertices_pred[:, :-1]).size()

            fluency_loss = fluency_loss * vertice_mask[:, 1:].unsqueeze(2) * change_location_mask.unsqueeze(2)
            fluency_loss = torch.sum(fluency_loss) / torch.sum(vertice_mask[:, 1:]) / vertices_gt.shape[2]

            basic_loss += fluency_loss
            self.log(f"{process}_fluency_loss", fluency_loss, prog_bar=True, batch_size=vertices_gt.shape[0])
        self.log(f"{process}_total_loss", basic_loss, prog_bar=True, batch_size=vertices_gt.shape[0])
        return basic_loss

