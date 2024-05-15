import os

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # egl
#
import numpy as np
import torch
import torch.optim as optim
from lightning import LightningModule

from config import cfg
from model import MamlTalk
from torch import nn
from render_utils import render_sequence_meshes
from utils import split_batch, length_same
from utils import mse_computation


# Temporal Bias, brrowed from https://github.com/EvelynFan/FaceFormer/blob/main/faceformer.py


class MamlTrainer(LightningModule):

    def __init__(self):
        super(MamlTrainer, self).__init__()
        self.lip_vertice_mapper = None
        self.automatic_optimization = False
        self.cfg = cfg
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

    def test_step(self, batch, batch_idx):
        # vertice_predictions, vertices, vertice_mask, match_vertice = self.get_self_prediction(batch)
        (audio, vertices, template, filenames,
         vertices_mask, speaker_ids) = batch
        vertice_predictions = self.model.forward(audio, vertices_mask
                                                 )
        self.loss_function(
            vertice_predictions, vertices, vertice_mask=vertices_mask, process='test'
        )
        batch_size = vertices.shape[0]
        motion_stds = []
        lip_dis_mouth_max = []
        for i in range(batch_size):
            vertice, vertice_prediction, vertice_ma = vertices[i], vertice_predictions[i], vertices_mask[i]
            motion_std_difference, L2_dis_mouth_max = mse_computation(vertice, vertice_prediction, self.model.upper_map,
                                                                      self.model.mouth_map, vertice_ma)
            motion_stds.append(motion_std_difference)
            lip_dis_mouth_max.append(L2_dis_mouth_max.cpu())
        self.log("motion_std_difference", np.mean(np.stack(motion_stds, axis=0)).item(), prog_bar=True,
                 batch_size=vertices.shape[0])
        self.lip_vertice_mapper.extend(lip_dis_mouth_max)
        self.log("lip_dis_mouth_max", torch.mean(torch.concatenate(lip_dis_mouth_max, dim=0)).cpu().item(),
                 prog_bar=True, batch_size=vertices.shape[0])

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
        (audio, vertice, template, filenames,
         vertices_masks, speaker_ids) = batch
        vertice_predictions = self.model.forward(audio, vertices_masks
                                                 )
        vertice_predictions = vertice_predictions + template
        # render predict vertices
        vertice_predictions = vertice_predictions.cpu().numpy()[0]

        input_audio_path = os.path.join(cfg.path.wav, filenames[0])
        file_type = input_audio_path.split('/')[-1].split('.')[0]
        render_sequence_meshes(
            input_audio_path, vertice_predictions, self.model.template_mesh, cfg.save_path,
            file_type=file_type + '_pred',
            ft=None,
            vt=None,
            tex_img=None
        )

        # render gt vertices
        render_sequence_meshes(
            input_audio_path, vertice.cpu().numpy()[0], self.model.template_mesh, cfg.save_path,
            file_type=file_type + '_gt', ft=None,
            vt=None, tex_img=None,
        )
        pred_file = os.path.join(cfg.save_path, file_type + '_pred.mp4')
        generated_file = os.path.join(cfg.save_path, file_type + '_gt.mp4')
        save_file = os.path.join(cfg.save_path, file_type + '_comb.mp4')
        os.system(
            f"ffmpeg -i {generated_file}-i {pred_file} -filter_complex '[0:0] [0:1] [1:0] [1:1] [2:0] [2:1] concat=n=3:v=1:a=1 [v] [a]' -map '[v]' -map '[a]’ {save_file}")

        return vertice_predictions

    def training_step(self, batch, batch_idx):
        (audio, vertice, template, filenames,
         vertices_masks, speaker_ids) = batch
        vertice_predictions = self.model.forward(audio, vertices_masks
                                                 )
        vertice_predictions = vertice_predictions + template
        loss = self.loss_function(
            vertice_predictions, vertice, vertice_mask=vertices_masks, process='train'
        )
        return loss
        # self.outer_loop(batch, mode="train")
        # return None  # Returning None means we skip the default training optimizer steps by PyTorch Lightning

    def validation_step(self, batch, batch_idx):
        # Validation requires to finetune a model, hence we need to enable gradients
        (audio, vertice, template, filenames,
         vertices_masks, speaker_ids) = batch
        vertice_predictions = self.model.forward(audio, vertices_masks
                                                 )
        vertice_predictions = vertice_predictions + template
        loss = self.loss_function(
            vertice_predictions, vertice, vertice_mask=vertices_masks, process='valid'
        )
        return loss
    def test_step(self, batch, batch_idx):
        # vertice_predictions, vertices, vertice_mask, match_vertice = self.get_self_prediction(batch)
        (audio, vertices, template, filenames,
         vertices_mask, speaker_ids) = batch
        vertice_predictions = self.model.forward(audio, vertices_mask
                                                 )
        self.loss_function(
            vertice_predictions, vertices, vertice_mask=vertices_mask, process='test'

        )
        batch_size = vertices.shape[0]
        motion_stds = []
        lip_dis_mouth_max = []
        for i in range(batch_size):
            vertice, vertice_prediction, vertice_ma = vertices[i], vertice_predictions[i], vertices_mask[i]
            motion_std_difference, L2_dis_mouth_max = mse_computation(vertice, vertice_prediction,
                                                                      self.model.upper_map,
                                                                      self.model.mouth_map, vertice_ma)
            motion_stds.append(motion_std_difference)
            lip_dis_mouth_max.append(L2_dis_mouth_max.cpu())
        self.log("motion_std_difference", np.mean(np.stack(motion_stds, axis=0)).item(), prog_bar=True,
                 batch_size=vertices.shape[0])
        self.lip_vertice_mapper.extend(lip_dis_mouth_max)
        self.log("lip_dis_mouth_max", torch.mean(torch.concatenate(lip_dis_mouth_max, dim=0)).cpu().item(),
                 prog_bar=True, batch_size=vertices.shape[0])
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
