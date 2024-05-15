import copy

from utils import mse_computation, length_same
from dataloader.retrain_dataloader import getTestDataset
from config import cfg
from torch.optim import Adam
import torch
from trainer import MamlTrainer
import numpy as np
from tqdm import tqdm
from args import get_args


def TrainSingleSample(trainer):
    test_dataset=getTestDataset()
    sample_dict=test_dataset.person_file_dict
    total_diff=[]
    total_lip_vertice_max=[]
    for speaker in sample_dict.keys():
        for sample_idx in range(len(sample_dict[speaker])):
            model=copy.deepcopy(trainer.model)
            model=model.cuda()
            device='cuda'
            optimizer = Adam(model.parameters(), lr=cfg.finetune_lr)

            for i in range(cfg.finetune_step):
                optimizer.zero_grad()
                sample=test_dataset.__getitem__((sample_idx,speaker))
                audio, vert, temp, speaker, _ = sample
                audio=audio.to(trainer.device).unsqueeze(0)
                vertice=vert.to(device).unsqueeze(0)
                temp=temp.to(device).unsqueeze(0)

                vertice_mask=torch.ones(vertice.shape[:2]).to(device)

                predict=model(audio,vertice_mask)
                vertice, vertice_prediction = length_same(vertice, predict)

                template=temp.view(temp.shape[0],1,-1)
                predict_face=predict+template
                loss=trainer.loss_function(vertice,predict_face,vertice_mask)
                # print(f'loss for step {i} is {loss}')
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                motion_stds = []
                lip_dis_mouth_max = []
                model.eval()
                for test_idx in range(len(sample_dict[speaker])):
                    if test_idx==sample_idx:
                        continue
                    sample = test_dataset.__getitem__((test_idx, speaker))
                    audio, vert, temp, speaker, _ = sample
                    audio,vert,temp=audio.to(device).unsqueeze(0),vert.to(device).unsqueeze(0),temp.to(device).unsqueeze(0)
                    vertice_mask=torch.ones(vert.shape[:2]).to(device)

                    vertice_prediction=model(audio,vertice_mask)
                    vertice_prediction=vertice_prediction+temp.view(1,1,-1)
                    vertice,vertice_prediction=length_same(vert,vertice_prediction)
                    vertice_mask,vertice_prediction=length_same(vertice_mask,vertice_prediction)

                    motion_std_difference, L2_dis_mouth_max = mse_computation(vertice, vertice_prediction,
                                                                              model.upper_map,
                                                                                model.mouth_map, vertice_mask)

                    motion_stds.append(motion_std_difference)
                    lip_dis_mouth_max.append(L2_dis_mouth_max.cpu())
                motion_std = np.mean(np.stack(motion_stds, axis=0))
                l2_dis_mouth_max = np.mean(np.concatenate(lip_dis_mouth_max, axis=0))
                print(f"for sentence {sample_idx} motion std is {motion_std},lip error is {l2_dis_mouth_max}")
                total_diff.append(motion_std)
                total_lip_vertice_max.append(l2_dis_mouth_max)
            # exit()
    print(f'mean total lip diff is {np.mean(total_lip_vertice_max)}')
    print(f"mean total std is {np.mean(total_diff)}")

if __name__=="__main__":
    model = MamlTrainer.load_from_checkpoint('/data3/zhouxukun/mamlface/result/NORMAL/pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio_/version_1/checkpoints/epoch=529-valid_total_loss=0.0003267751.ckpt')

    #.load_from_checkpoint(
   #     '/data3/zhouxukun/mamlface/result/pretrainedNone_crosstrainFalse_usetransformerFalse_backbonehubert_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio_/version_1/checkpoints/epoch=929-valid_total_loss=0.0003739640.ckpt')
    TrainSingleSample(model
                      )









