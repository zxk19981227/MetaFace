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
from transformers import AutoProcessor
from args import get_args
import torchaudio
import pickle
@torch.no_grad()
def PredictSample(model,cfg,name,audio_path,template_name):
    speech_array, sampling_rate = torchaudio.load(audio_path)
    sampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    speech_array = sampler(speech_array)
    # preprocessor_path = cfg.path.wav2vec

    preprocessor_path='/data3/zhouxukun/transformer_model/wav2vec2-large-960h-lv60-self'
    with open(cfg.path.template, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')
    temp=templates[template_name]
    processor = AutoProcessor.from_pretrained(
        preprocessor_path
    )
    input_values = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
    input_values=torch.from_numpy(input_values).cuda()
    vertice_prediction = model(input_values, torch.ones(input_values.shape[:2]).cuda())[0]
    vertice_prediction = vertice_prediction.cpu().numpy() + temp.reshape(1, 1, -1)
    audio_file_name=audio_path.split('/')[-1].replace('.mp3','')

    file_type=f"{audio_file_name}_speaker_{name}_template_{template_name}.mp4"
    render_sequence_meshes(
        audio_path=audio_path, sequence_vertices=vertice_prediction[0],
        template=model.template_mesh, out_path='/data3/zhouxukun/final_save',
        file_type=file_type, ft=None,
        vt=None, tex_img=None
    )



if __name__=="__main__":
    from args import get_args
    args=get_args()
    config_file = args.cfg

    cfg.merge_from_file(config_file)
    cfg=update_cfg(cfg)
    model=MamlTrainer.load_from_checkpoint(
        # "result_backup/way11_f1_pretrainedNone_crosstrainFalse_usetransasfformerFalse_backbonewav2vec_0.0001lr_1batch_size_BIWIdataset_1024feature_dim_70110vertice_dimFalsefreeze_audio__lora_neural/version_1/checkpoints/epoch=896-valid_total_loss=0.0178567581.ckpt")
    # )
        'result/way11_f1_pretrainedNone_crosstrainFalse_usetransformerFalse_backbonewav2vec_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio__lora_neural/version_1/checkpoints/epoch=881-valid_total_loss=0.0003616871.ckpt')
        # '/data3/zhouxukun/mamlface/result/pretrainedNone_crosstrainFalse_usetransformerFalse_backbonehubert_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio_/version_1/checkpoints/epoch=929-valid_total_loss=0.0003739640.ckpt')
# )
#     model.model.load_state_dict(torch.load('/data3/zhouxukun/mamlface/speaker_checkpoints/FaceTalk_170809_00138_TA.pth'))
    name="FaceTalk_170731_00024_TA"
    # name='FaceTalk_170809_00138_TA'
    # name='F7'
    model.model.load_state_dict(torch.load(f'/data3/zhouxukun/mamlface/speaker_checkpoints/{name}.pth'))
    audio_path='/data3/zhouxukun/split_audio/'
    audio_files=os.listdir(audio_path)
    for audio_file in audio_files:
        audio_file=os.path.join(audio_path,audio_file)
        PredictSample(model.model.cuda().eval(),cfg,name,audio_file,template_name=name)





