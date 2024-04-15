import os
import pickle
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from numpy.linalg import norm

from utils import length_same, split_batch
from render_utils import render_sequence_meshes
from modules.dtw import dtw
from lightning import LightningModule
from modules.hubert import Hubert2Vec
from modules.wave2vec import Wav2Vec2Model
from config import cfg
from utils import mse_computation
import torch.optim as optim

from psbody.mesh import Mesh


# Temporal Bias, brrowed from https://github.com/EvelynFan/FaceFormer/blob/main/faceformer.py


class MabaTalk(LightningModule):

    def __init__(self):
        super(MabaTalk, self).__init__()
        self.save_hyperparameters()
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
            self, audio, template, audio_mask
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

        vertice_out = vertice_out + template
        return vertice_out

    def loss_function(
            self, vertices_gt, vertices_pred, vertice_mask, change_location=None, process='train',
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

    def training_step(self, batch, batch_idx):
        vertice_prediction, vertice, vertices_masks, speaker_ids= self.get_self_prediction(batch)
        vertice_prediction, vertice, vertices_masks, speaker_ids= split_batch(features, template, vertices_masks,speaker_ids)
        prototypes, classes = self.calculate_prototypes(support_feats, support_targets)
        preds, labels, acc = self.classify_feats(prototypes, classes, query_feats, query_targets)
        loss = F.cross_entropy(preds, labels)
        loss = self.loss_function(
            vertice, vertice_prediction, vertice_mask, 'train'
        )

        return loss
    @staticmethod
    def calculate_prototypes(features, targets):
        # Given a stack of features vectors and labels, return class prototypes
        # features - shape [N, proto_dim], targets - shape [N]
        classes, _ = torch.unique(targets).sort()  # Determine which classes we have
        prototypes = []
        for c in classes:
            p = features[torch.where(targets == c)[0]].mean(dim=0)  # Average class feature vectors
            prototypes.append(p)
        prototypes = torch.stack(prototypes, dim=0)
        # Return the 'classes' tensor to know which prototype belongs to which class
        return prototypes, classes

    def get_self_prediction(self, batch):
        (audio, vertice, template, filenames, audio_masks,
         vertices_masks, speaker_ids) = batch
        vertice_prediction = self.forward(
            audio, template, audio_masks
        )

        return vertice_prediction, vertice, vertices_masks,speaker_ids

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[140, 180], gamma=0.1)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        vertice_prediction, vertice, vertice_mask, match_vertice, change_location = self.get_self_prediction(batch)

        self.loss_function(vertice_prediction, vertice, vertice_mask, change_location, 'valid')

    def on_test_epoch_start(self) -> None:
        self.lip_vertice_mapper = []

    def on_test_epoch_end(self) -> None:
        lip_vertice_mapper = torch.concatenate(self.lip_vertice_mapper, dim=0)
        lip_vertice_mapper = np.mean(lip_vertice_mapper.cpu().numpy())
        self.log('lip_vertice_map', lip_vertice_mapper, prog_bar=True)

    def test_step(self, batch, batch_idx):
        vertice_prediction, vertice, vertices_masks,speaker_ids = self.get_self_prediction(batch)

        self.loss_function(
            vertice_predictions, vertices, vertice_mask=vertice_mask, process='test'
            # , match_audio_vertices=match_vertice
        )
        batch_size = vertices.shape[0]
        motion_stds = []
        lip_dis_mouth_max = []
        for i in range(batch_size):
            vertice, vertice_prediction, vertice_ma = vertices[i], vertice_predictions[i], vertice_mask[i]
            motion_std_difference, L2_dis_mouth_max = mse_computation(vertice, vertice_prediction, self.upper_map,
                                                                      self.mouth_map, vertice_ma)
            motion_stds.append(motion_std_difference)
            lip_dis_mouth_max.append(L2_dis_mouth_max.cpu())
        self.log("motion_std_difference", np.mean(np.stack(motion_stds, axis=0)).item(), prog_bar=True,
                 batch_size=vertices.shape[0])
        self.lip_vertice_mapper.extend(lip_dis_mouth_max)
        self.log("lip_dis_mouth_max", torch.mean(torch.concatenate(lip_dis_mouth_max, dim=0)).cpu().item(),
                 prog_bar=True, batch_size=vertices.shape[0])

    def predict_step(self, batch, batch_idx):

        if not os.path.exists(cfg.path.save):
            os.mkdir(cfg.path.save)
        # (audio, vertice, template, filenames,reference_name, reference_motion, reference_audio, audio_mask,
        # vertice_mask, match_audio_vertice,reference_audio_mask, reference_motion_mask) = batch
        audio, vertice, template, filenames, reference_name, reference_motion, reference_audio, audio_mask, vertice_mask, reference_audio_mask, reference_motion_mask, ref_poho, gt_pho = batch

        vertice_predictions = self.forward(audio, template, audio_mask)

        # render predict vertices
        vertice_predictions = vertice_predictions.cpu().numpy()[0]

        input_audio_path = os.path.join(cfg.path.wav, filenames[0])
        file_type = input_audio_path.split('/')[-1].split('.')[0]
        render_sequence_meshes(
            input_audio_path, vertice_predictions, self.template_mesh, cfg.save_path, file_type=file_type + '+pred',
            ft=None,
            vt=None,
            tex_img=None
        )

        # render gt vertices
        render_sequence_meshes(
            input_audio_path, vertice.cpu().numpy()[0], self.template_mesh, cfg.save_path,
            file_type=file_type + '_gt_regenerated', ft=None,
            vt=None, tex_img=None,
        )

        return vertice_predictions



    def adapt_few_shot(self, support_imgs, support_targets):
        # Determine prototype initialization
        support_feats = self.model(support_imgs)
        prototypes, classes = self.calculate_prototypes(support_feats, support_targets)
        support_labels = (classes[None, :] == support_targets[:, None]).long().argmax(dim=-1)
        # Create inner-loop model and optimizer
        local_model = deepcopy(self.model)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), lr=self.hparams.lr_inner)
        local_optim.zero_grad()
        # Create output layer weights with prototype-based initialization
        init_weight = 2 * prototypes
        init_bias = -torch.norm(prototypes, dim=1) ** 2
        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()

        # Optimize inner loop model on support set
        for _ in range(self.hparams.num_inner_steps):
            # Determine loss on the support set
            loss, _, _ = self.run_model(local_model, output_weight, output_bias, support_imgs, support_labels)
            # Calculate gradients and perform inner loop update
            loss.backward()
            local_optim.step()
            # Update output layer via SGD
            output_weight.data -= self.hparams.lr_output * output_weight.grad
            output_bias.data -= self.hparams.lr_output * output_bias.grad
            # Reset gradients
            local_optim.zero_grad()
            output_weight.grad.fill_(0)
            output_bias.grad.fill_(0)

        # Re-attach computation graph of prototypes
        output_weight = (output_weight - init_weight).detach() + init_weight
        output_bias = (output_bias - init_bias).detach() + init_bias

        return local_model, output_weight, output_bias, classes

    def outer_loop(self, batch, mode="train"):
        accuracies = []
        losses = []
        self.model.zero_grad()

        # Determine gradients for batch of tasks
        for task_batch in batch:
            imgs, targets = task_batch
            support_imgs, query_imgs, support_targets, query_targets = split_batch(imgs, targets)
            # Perform inner loop adaptation
            local_model, output_weight, output_bias, classes = self.adapt_few_shot(support_imgs, support_targets)
            # Determine loss of query set
            query_labels = (classes[None, :] == query_targets[:, None]).long().argmax(dim=-1)
            loss, preds, acc = self.run_model(local_model, output_weight, output_bias, query_imgs, query_labels)
            # Calculate gradients for query set loss
            if mode == "train":
                loss.backward()

                for p_global, p_local in zip(self.model.parameters(), local_model.parameters()):
                    p_global.grad += p_local.grad  # First-order approx. -> add gradients of finetuned and base model

            accuracies.append(acc.mean().detach())
            losses.append(loss.detach())

        # Perform update of base model
        if mode == "train":
            opt = self.optimizers()
            opt.step()
            opt.zero_grad()

        self.log("%s_loss" % mode, sum(losses) / len(losses))
        self.log("%s_acc" % mode, sum(accuracies) / len(accuracies))

    def training_step(self, batch, batch_idx):
        self.outer_loop(batch, mode="train")
        return None  # Returning None means we skip the default training optimizer steps by PyTorch Lightning

    def validation_step(self, batch, batch_idx):
        # Validation requires to finetune a model, hence we need to enable gradients
        torch.set_grad_enabled(True)
        self.outer_loop(batch, mode="val")
        torch.set_grad_enabled(False)