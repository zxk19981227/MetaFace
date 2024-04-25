import os
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from lightning import LightningModule

from config import cfg
from model import MamlTalk
from torch import nn
from render_utils import render_sequence_meshes
from utils import split_batch, length_same


# Temporal Bias, brrowed from https://github.com/EvelynFan/FaceFormer/blob/main/faceformer.py


class MamlTrainer(LightningModule):

    def __init__(self):
        super(MamlTrainer, self).__init__()
        self.lip_vertice_mapper = None
        self.automatic_optimization=False
        self.cfg=cfg
        self.save_hyperparameters()
        self.model = MamlTalk()
        self.mse_func = nn.MSELoss(reduction='none')

    def get_self_prediction(self, batch):
        (audio, vertice, template, filenames, audio_masks,
         vertices_masks, speaker_ids) = batch
        vertice_prediction = self.model(
            audio, template, audio_masks
        )
        return vertice_prediction, vertice, vertices_masks, speaker_ids

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[140, 180], gamma=0.1)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        vertice_prediction, vertice, vertice_mask, match_vertice, change_location = self.get_self_prediction(batch)

        self.loss_function(vertice_prediction, vertice, vertice_mask, change_location, 'valid')

    def adapt_few_shot(self, support_audios, support_vertice_mask, support_vertice, support_speaker):
        # Determine prototype initialization
        # print(support_vertice_mask.shape)
        # exit()
        # support_pred = self.model(support_audios, support_vertice_mask)
        #  这里原本是说，输入一个文本以后把这个support和targets的protype进行分类，但是实际做regression任务并且不实用prototype时候这里完全没有用
        # support_labels = (classes[None, :] == support_targets[:, None]).long().argmax(dim=-1)
        # Create inner-loop model and optimizer
        local_model = deepcopy(self.model)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), lr=self.cfg.lr_inner)
        local_optim.zero_grad()
        # Create output layer weights with prototype-based initialization
        # 这里原本是用来初始化相关的模型代码的。但是不用prototypes完全不用这个

        # Optimize inner loop model on support set
        for _ in range(self.cfg.num_inner_steps):
            # Determine loss on the support set
            predict_vertice = local_model(support_audios, support_vertice_mask)
            loss=self.loss_function(predict_vertice, support_vertice, support_vertice_mask)
            # Calculate gradients and perform inner loop update
            loss.backward()
            local_optim.step()
            local_optim.zero_grad()

        # Re-attach computation graph of prototypes
        # output_weight = (output_weight - init_weight).detach() + init_weight
        # output_bias = (output_bias - init_bias).detach() + init_bias

        return local_model

    def outer_loop(self, batch, mode="train"):
        accuracies = []
        losses = []
        self.model.zero_grad()

        # Determine gradients for batch of tasks
        for task_batch in batch:
            (
                audio, vertice, vertices_mask,template, speaker_ids
            ) = task_batch  # Perform inner loop adaptation
            (
                support_audio, query_audio, support_vertice, query_vertice, support_vertice_mask,
                query_vertice_mask, support_speaker, query_speaker
            ) = split_batch(audio, vertice, vertices_mask, speaker_ids)

            local_model = self.adapt_few_shot(
                support_audios=support_audio,support_vertice_mask=support_vertice_mask,support_vertice=support_vertice,
                support_speaker=support_speaker
            )
            # Determine loss of query set
            # query_labels = (classes[None, :] == query_targets[:, None]).long().argmax(dim=-1)
            pred_vertices=local_model(query_audio,query_vertice_mask)
            loss = self.loss_function(pred_vertices, query_vertice,query_vertice_mask,mode)
            # Calculate gradients for query set loss
            if mode == "train":
                loss.backward()
                for p_global, p_local in zip(self.model.named_parameters(), local_model.named_parameters()):
                    if p_local[1].grad is None:
                        continue
                    if p_global[1].grad is None and p_local[1].grad is not None:
                        p_global[1].grad=torch.zeros_like(p_local[1].grad)
                    p_global[1].grad += p_local[1].grad  # First-order approx. -> add gradients of finetuned and base mode
            losses.append(loss.detach())

        # Perform update of base model
        if mode == "train":
            opt = self.optimizers()
            opt.step()
            opt.zero_grad()

        self.log("%s_loss" % mode, sum(losses) / len(losses))

    def loss_function(
            self, vertices_gt, vertices_pred, vertice_mask, process='train',
    ):

        vertices_pred, vertices_gt = length_same(vertices_pred, vertices_gt)
        vertice_mask, vertices_gt = length_same(vertice_mask, vertices_gt)
        # change_location_mask = torch.abs(change_location[:, 1:] - change_location[:, :-1])

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

            fluency_loss = fluency_loss * vertice_mask[:, 1:].unsqueeze(2)  # * change_location_mask.unsqueeze(2)
            fluency_loss = torch.sum(fluency_loss) / torch.sum(vertice_mask[:, 1:]) / vertices_gt.shape[2]

            basic_loss += fluency_loss
            self.log(f"{process}_fluency_loss", fluency_loss, prog_bar=True, batch_size=vertices_gt.shape[0])
        self.log(f"{process}_total_loss", basic_loss, prog_bar=True, batch_size=vertices_gt.shape[0])
        return basic_loss
    def on_test_epoch_start(self) -> None:
        self.lip_vertice_mapper = []

    def on_test_epoch_end(self) -> None:
        lip_vertice_mapper = torch.concatenate(self.lip_vertice_mapper, dim=0)
        lip_vertice_mapper = np.mean(lip_vertice_mapper.cpu().numpy())
        self.log('lip_vertice_map', lip_vertice_mapper, prog_bar=True)



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

    # def adapt_few_shot(self, support_imgs, support_targets):
    #     # Determine prototype initialization
    #     support_feats = self.model(support_imgs)
    #     prototypes, classes = self.calculate_prototypes(support_feats, support_targets)
    #     support_labels = (classes[None, :] == support_targets[:, None]).long().argmax(dim=-1)
    #     # Create inner-loop model and optimizer
    #     local_model = deepcopy(self.model)
    #     local_model.train()
    #     local_optim = optim.SGD(local_model.parameters(), lr=self.hparams.lr_inner)
    #     local_optim.zero_grad()
    #     # Create output layer weights with prototype-based initialization
    #     init_weight = 2 * prototypes
    #     init_bias = -torch.norm(prototypes, dim=1) ** 2
    #     output_weight = init_weight.detach().requires_grad_()
    #     output_bias = init_bias.detach().requires_grad_()
    #
    #     # Optimize inner loop model on support set
    #     for _ in range(self.hparams.num_inner_steps):
    #         # Determine loss on the support set
    #         loss, _, _ = self.run_model(local_model, output_weight, output_bias, support_imgs, support_labels)
    #         # Calculate gradients and perform inner loop update
    #         loss.backward()
    #         local_optim.step()
    #         # Update output layer via SGD
    #         output_weight.data -= self.hparams.lr_output * output_weight.grad
    #         output_bias.data -= self.hparams.lr_output * output_bias.grad
    #         # Reset gradients
    #         local_optim.zero_grad()
    #         output_weight.grad.fill_(0)
    #         output_bias.grad.fill_(0)
    #
    #     # Re-attach computation graph of prototypes
    #     output_weight = (output_weight - init_weight).detach() + init_weight
    #     output_bias = (output_bias - init_bias).detach() + init_bias
    #
    #     return local_model, output_weight, output_bias, classes

    def training_step(self, batch, batch_idx):
        self.outer_loop(batch, mode="train")
        # return None  # Returning None means we skip the default training optimizer steps by PyTorch Lightning

    def validation_step(self, batch, batch_idx):
        # Validation requires to finetune a model, hence we need to enable gradients
        torch.set_grad_enabled(True)
        self.outer_loop(batch, mode="valid")
        torch.set_grad_enabled(False)
    # @torch.no_grad()
    # def test_proto_net(self, dataset,):
    #     self.model.eval()
    #     num_classes = dataset.targets.unique().shape[0]
    #     exmps_per_class = [torch.sum(dataset[dataset["target"] ==i]) for i in dataset.targets.unique()]
    #     k_shot=self.k_shot
    #     # The encoder network remains unchanged across k-shot settings. Hence, we only need
    #     # to extract the features for all images once.
    #
    #
    #     return (mean(accuracies), stdev(accuracies)), (img_features, img_targets)



