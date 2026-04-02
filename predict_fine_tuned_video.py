import copy

from utils import mse_computation, length_same, update_cfg
from dataloader.retrain_dataloader import getTestDataset
from config import cfg
from torch.optim import Adam
import torch
from trainer import MamlTrainer
import numpy as np
from render_utils import render_sequence_meshes
import os
from tqdm import tqdm
from args import get_args

def TrainSingleSample(trainer,speaker_name):
    test_dataset=getTestDataset()
    retrain_dataset=getTestDataset('test',usage='train')
    sample_dict=test_dataset.person_file_dict
    sample_number=sample_dict[speaker_name]
    total_diff=[]
    total_lip_vertice_max=[]
    # sample_speaker=list(sample_dict.keys())[0]
    sample_idx=2
    test_idx=1
    model=copy.deepcopy(trainer.model)
    model=model.cuda()
    dataset_samples=[4,5,6,7]

    device='cuda'
    optimizer = Adam(model.parameters(), lr=cfg.finetune_lr)

    for i in tqdm(range(200)):
        for idx in range(4):
            optimizer.zero_grad()
            sample=retrain_dataset.__getitem__((dataset_samples[i],speaker_name))
            audio, vert, temp, speaker, _ = sample
            audio=audio.to(device).unsqueeze(0)
            vertice=vert.to(device).unsqueeze(0)
            temp = temp.to(device).unsqueeze(0)
            vertice_mask=torch.ones(vertice.shape[:2]).to(device)

            predict=model(audio,vertice_mask)[0]
            vertice, vertice_prediction = length_same(vertice, predict)

            template=temp.reshape(temp.shape[0],1,-1)
            predict_face=predict+template
            loss=trainer.loss_function(vertice,predict_face,vertice_mask)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        model.eval()
        motion_stds = []
        lip_dis_mouth_max = []
        model.eval()
        for test_idx in range(len(sample_number)):
            sample = test_dataset.__getitem__((test_idx, speaker_name))
            audio, vert, temp, speaker, file_name = sample
            audio,vert,temp=audio.to(device).unsqueeze(0),vert.to(device).unsqueeze(0),temp.to(device).unsqueeze(0)
            vertice_mask=torch.ones(vert.shape[:2]).to(device)

            vertice_prediction=model(audio,vertice_mask)[0]
            vertice_prediction=vertice_prediction+temp.reshape(1,1,-1)
            vertice,vertice_prediction=length_same(vert,vertice_prediction)
            vertice_mask,vertice_prediction=length_same(vertice_mask,vertice_prediction)

            motion_std_difference, L2_dis_mouth_max = mse_computation(vertice, vertice_prediction,
                                                                      model.upper_map,
                                                                        model.mouth_map, vertice_mask)

            motion_stds.append(motion_std_difference)
            lip_dis_mouth_max.append(L2_dis_mouth_max.cpu())
            # print(vertice_prediction e.shape)
            render_sequence_meshes(
                audio_path=os.path.join(cfg.path.wav,file_name),sequence_vertices=vertice_prediction.cpu().numpy()[0],
                template=model.template_mesh, out_path=cfg.save_path,
                file_type=file_name.replace('.mp4','') + '_pred', ft=None,
                vt=None, tex_img=None,
            )
            render_sequence_meshes(
                audio_path=os.path.join(cfg.path.wav,file_name),sequence_vertices=vertice.cpu().numpy()[0],
                template=model.template_mesh, out_path=cfg.save_path,
                file_type=file_name.replace('.mp4','') + '_gt', ft=None,
                vt=None, tex_img=None,
                                   )
    print(f'mean total lip diff is {np.mean(total_lip_vertice_max)}')
    print(f"mean total std is {np.mean(total_diff)}")

if __name__=="__main__":
    from args import get_args
    args=get_args()
    config_file = args.cfg

    cfg.merge_from_file(config_file)
    cfg=update_cfg(cfg)
    model=MamlTrainer.load_from_checkpoint(
        "result_backup/way11_f1_pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_BIWIdataset_1024feature_dim_70110vertice_dimFalsefreeze_audio__lora_neural/version_1/checkpoints/epoch=896-valid_total_loss=0.0178567581.ckpt")
        #'result/way11_f1_pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio__lora_neural/version_1/checkpoints/epoch=881-valid_total_loss=0.0003616871.ckpt')
    #     '/data3/zhouxukun/mamlface/result/pretrainedNone_crosstrainFalse_usetransformerFalse_backbonehubert_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio_/version_1/checkpoints/epoch=929-valid_total_loss=0.0003739640.ckpt')
    for speaker in cfg.train.test_subjects.split(' '):
        TrainSingleSample(
            model,speaker
        )









