import copy
import random

from tqdm import tqdm

from args import get_args
from utils import mse_computation, length_same
from dataloader.retrain_dataloader import getTestDataset
from config import cfg
from torch.optim import Adam
import torch
from trainer import MamlTrainer
import numpy as np
import os
from minlora import add_lora, apply_to_lora, disable_lora, enable_lora, get_lora_params, merge_lora, name_is_lora, remove_lora, load_multiple_lora, select_lora
from utils import Metric,update_cfg

def TrainSingleSample(trainer):
    test_dataset=getTestDataset()
    train_dataset=getTestDataset('test','train')
    train_sample_dict=train_dataset.person_file_dict
    sample_dict=test_dataset.person_file_dict
    assert train_sample_dict.keys()==sample_dict.keys()
    total_face_l2=[]
    total_lip_l2=[]
    total_lip_dtw=[]
    total_lip_l2_max=[]
    total_lip_vertice_max=[]
    total_diff=[]
    metrics=Metric()

    for speaker in tqdm(sample_dict.keys()):

        sample_idxs=random.sample(list(range(len(train_sample_dict[speaker]))),cfg.n_samples)
        model=copy.deepcopy(trainer.model).cpu()
        # remove_lora(model)
        # add_lora(model)
        device='cuda'

        model=model.cuda()
        refined_params=[]
        for name,param in model.named_parameters():
            if 'vertice_remap' in name:
                refined_params.append(param)

        parameters = [
            {"params": list(get_lora_params(model))+refined_params},
            # {"params": list(model.parameters())}
        ]
        total = sum([param.nelement() for param in parameters[0]['params'] if param.requires_grad])
        print(f'total param for pre is {total/1e6}')
        # for param in model.parameters():
        #     if param not in parameters[0]['params']:
        #         param.requires_grad=False
        # parameters=model.parameters()
        # optimizer = torch.optim.AdamW(parameters, lr=1e-5)
        optimizer = Adam(parameters, lr=cfg.finetune_lr)

        for i in range(cfg.finetune_step):
            random.shuffle(sample_idxs)
            for sample_idx in sample_idxs:
                optimizer.zero_grad()
                sample=train_dataset.__getitem__((sample_idx,speaker))
                audio, vert, temp, speaker, _ = sample
                audio=audio.to(device).unsqueeze(0)
                vertice=vert.to(device).unsqueeze(0)
                temp=temp.to(device).unsqueeze(0)

                vertice_mask=torch.ones(vertice.shape[:2]).to(device)
                if model.neural_process:
                    predict,_=model(audio,vertice_mask)
                else:
                    predict=model(audio,vertice_mask)
                vertice, vertice_prediction = length_same(vertice, predict)

                template=temp.reshape(temp.shape[0],1,-1)
                predict_face=predict+template
                loss=trainer.loss_function(vertice,predict_face,vertice_mask)
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(),os.path.join('speaker_checkpoints',f"{speaker}.pth"))
        with torch.no_grad():
            motion_stds = []
            lip_dis_mouth_max = []
            model.eval()
            for test_idx in range(len(sample_dict[speaker])):
                sample = test_dataset.__getitem__((test_idx, speaker))
                audio, vert, temp, speaker, _ = sample
                audio,vert,temp=audio.to(device).unsqueeze(0),vert.to(device).unsqueeze(0),temp.to(device).unsqueeze(0)
                vertice_mask=torch.ones(vert.shape[:2]).to(device)

                vertice_prediction=model(audio,vertice_mask)
                vertice_prediction=vertice_prediction+temp.reshape(1,1,-1)

                vertice,vertice_prediction=length_same(vert,vertice_prediction)
                vertice_mask,vertice_prediction=length_same(vertice_mask,vertice_prediction)

                motion_std_difference, L2_dis_mouth_max = mse_computation(vertice, vertice_prediction,
                                                                          model.upper_map,
                                                                            model.mouth_map, vertice_mask)
                vertice=vertice[0]
                pose_length=vertice.shape[0]
                vertice_prediction=vertice_prediction[0].reshape(pose_length,-1,3)
                vertice=vertice.reshape(pose_length,-1,3)
                l2_face=metrics.l2_face(vertice,vertice_prediction)
                l2_lip=metrics.l2_lip(vertice,vertice_prediction)
                dtw_face=metrics.dtw_lip(vertice_prediction,vertice)
                max_lip=metrics.lip_max_l2(vertice,vertice_prediction)
                # lip_sync=metric.lip_sync(vertice_prediction,vertice)
                total_face_l2.append(l2_face.cpu())
                total_lip_l2.append(l2_lip.cpu())
                total_lip_dtw.append(dtw_face)
                total_lip_l2_max.append(max_lip.cpu())
                motion_stds.append(motion_std_difference)
                lip_dis_mouth_max.append(L2_dis_mouth_max.cpu())

            motion_std = np.mean(np.stack(motion_stds, axis=0))
            l2_dis_mouth_max = np.mean(np.concatenate(lip_dis_mouth_max, axis=0))
            total_diff.append(motion_std)
            total_lip_vertice_max.append(l2_dis_mouth_max)
        # exit()
    print(f'mean total lip diff is {np.mean(total_lip_vertice_max)}')
    # print(f"mean total std is {np.mean(total_diff)}")
    print(f"total face l2 is {np.mean(total_face_l2)}")
    print(f'total lip dw is {np.mean(total_lip_dtw)}')
    print(f"total lip l2 is {np.mean(total_lip_l2)}")
    print(f"total lip max is {np.mean(total_lip_l2_max)}")
if __name__=="__main__":
    args = get_args()
    config_file = args.cfg
    cfg.merge_from_file(config_file)
    update_cfg(cfg)

    model = MamlTrainer.load_from_checkpoint(
                # "result/way11_f1_pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio_/version_1/checkpoints/epoch=992-valid_total_loss=0.0003729885.ckpt"

        "result/way11_f1_pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio__lora_neural/version_1/checkpoints/epoch=959-valid_total_loss=0.0003474369.ckpt"
       # "result/way11_f1_pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_BIWIdataset_1024feature_dim_70110vertice_dimFalsefreeze_audio__neural/version_1/checkpoints/epoch=644-valid_total_loss=0.0186477359.ckpt"
    # "result_backup/way11_f1_pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_BIWIdataset_1024feature_dim_70110vertice_dimFalsefreeze_audio__lora_neural/version_1/checkpoints/epoch=497-valid_total_loss=0.0173654165.ckpt"
   # "result/pretrainedNone_crosstrainFalse_usetransformerFalse_backbonehubert_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio_/version_1/checkpoints/epoch=878-valid_total_loss=0.0073078154.ckpt"
   # "result/way11_f1_pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio__lora_neural/version_1/checkpoints/epoch=959-valid_total_loss=0.0003474369.ckpt"
   # 'result/way11_f1_pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio_/version_1/checkpoints/epoch=980-valid_total_loss=0.0003320088.ckpt'
   #  'result_backup/pretrainedNone_crosstrainFalse_usetransformerFalse_backbonehubert_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio__lora/version_1/checkpoints/epoch=569-valid_total_loss=0.0003348832.ckpt'
   # '/data3/zhouxukun/mamlface/result_backup/NORMAL/pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio_/version_1/checkpoints/epoch=690-valid_total_loss=0.0003373701.ckpt'
   # "result_backup/way3_f1_pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio__lora/version_1/checkpoints/epoch=434-valid_total_loss=0.0003803781.ckpt"
   # "result_backup/way8_f1_pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio__lora/version_1/checkpoints/epoch=953-valid_total_loss=0.0003621964.ckpt"
   #  "result/way11_f1_pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio__lora_neural/version_1/checkpoints/epoch=959-valid_total_loss=0.0003474369.ckpt"
   #  "result/way11_f1_pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio__lora/version_1/checkpoints/epoch=773-valid_total_loss=0.0003174071.ckpt"
   #  "result/normal/version_1/checkpoints/epoch=905-valid_total_loss=0.0003400870.ckpt"
   #      "result_backup/way3_f1_pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio__lora/version_1/checkpoints/epoch=281-valid_total_loss=0.0003810217.ckpt"
   # "result_backup/pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio__lora_neural/version_1/checkpoints/epoch=939-valid_total_loss=0.0003184178.ckpt"
    # "result_backup/NORMAL/pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio_/version_1/checkpoints/epoch=529-valid_total_loss=0.0003267751.ckpt"
    # "result_backup/way5_f2_pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio__lora/version_1/checkpoints/epoch=950-valid_total_loss=0.0003991832.ckpt"
    # "/data3/zhouxukun/mamlface/result_backup/way5_f3_pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio__lora/version_1/checkpoints/epoch=935-valid_total_loss=0.0004021995.ckpt"
   # "result_backup/way3_f1_pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio__lora/version_1/checkpoints/epoch=434-valid_total_loss=0.0003803781.ckpt"
    )
    TrainSingleSample(
        model
    )









