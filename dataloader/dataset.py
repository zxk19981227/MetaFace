import os
import pickle


import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from dataloader.dataset_utils import TaskBatchSampler



from config import cfg
from utils import check_path_valid, get_data_path, collate_fn


class AudioDataset(Dataset):
    def __init__(self, data_type):
        super().__init__()
        self.dataset_name = cfg.dataset
        self.backbone = cfg.backbone
        self.data_type = data_type
        self.splits = {'vocaset': {'train': range(1, 41), 'val': range(21, 41), 'test': range(21, 41)},
                       'BIWI': {'train': range(1, 33), 'val': range(33, 37), 'test': range(37, 41)}}
        self.data = []
        self.speaker_id=[]
        if not cfg.use_pregenerated_feature:

            if self.backbone == 'hubert':
                preprocessor_path = cfg.path.hubert
            elif self.backbone.lower() == 'wav2vec':
                preprocessor_path = cfg.path.wav2vec
            else:
                raise NotImplementedError('backbone {} not implemented'.format(self.backbone))
            self.processor = AutoProcessor.from_pretrained(
                preprocessor_path
            )
        else:
            self.processor = None
        self.audio_path_list, self.subjects, self.speaker_dict = get_data_path(dataset_type=data_type,
                                                                               split=self.splits[self.dataset_name])
        self.audio_path = cfg.path.wav
        self.vertices_path = cfg.path.vertices
        self.template_file = cfg.path.template
        self.sentences_id = self.splits[self.dataset_name]

        # group file by speaker
        self.person_file_dict = {}
        with open(self.template_file, 'rb') as fin:
            self.templates = pickle.load(fin, encoding='latin1')

        for audio_file_path in self.audio_path_list:
            if check_path_valid(audio_file_path, self.vertices_path):
                self.data.append(audio_file_path)
        for file in self.data:
            subject_id = '_'.join(file.split('_')[:-1])
            self.speaker_id.append(self.speaker_dict[subject_id])
            if subject_id not in self.person_file_dict.keys():
                self.person_file_dict[subject_id] = [file]
            else:
                self.person_file_dict[subject_id].append(file.replace("wav", "npy"))
        for file in self.data:
            user_name = '_'.join(file.split('_')[:4])
            if user_name not in self.person_file_dict:
                self.person_file_dict[user_name] = len(self.person_file_dict)
        self.speaker_id=torch.from_numpy(np.array(self.speaker_id))
        self.len = len(self.data)

    def load_audio(self, wav_path):
        if self.processor is None:
            input_values = np.load(
                os.path.join(cfg.path.audio_feature, wav_path).replace(".wav", f"_{cfg.backbone}.npy")
            )
        else:
            speech_array, sampling_rate = torchaudio.load(os.path.join(cfg.path.wav, wav_path))
            sampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            speech_array = sampler(speech_array)
            input_values = np.squeeze(self.processor(speech_array, sampling_rate=16000).input_values)
        return input_values
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        file_name = self.data[index]
        subject_id = '_'.join(file_name.split('_')[:-1])
        vertices_name = file_name.replace('wav', 'npy')

        audio = self.load_audio(file_name)
        vertices_path = os.path.join(self.vertices_path, vertices_name)
        if self.dataset_name == "vocaset":
            vertice = np.load(vertices_path, allow_pickle=True)[::2, :]

        elif self.dataset_name == "BIWI":
            vertice = np.load(vertices_path, allow_pickle=True)

        else:
            raise NotImplementedError('dataset{} not implemented'.format(self.dataset_name))
        speaker = '_'.join(vertices_name.split('_')[:4])
        if speaker not in self.speaker_dict.keys():
            raise NotImplementedError(f'{speaker} not exists in {self.speaker_dict.keys()} ')
        else:
            speaker_id = self.speaker_dict[speaker]
        template = self.templates[subject_id]
        return torch.from_numpy(audio),torch.from_numpy(vertice), torch.from_numpy(template),speaker_id,file_name

def get_dataloaders(batch_size=1):
    dataloaders = {}

    train_dataset=AudioDataset("train")
    train_sampler=TaskBatchSampler(
            train_dataset.speaker_id, include_query=True, N_way=cfg.n_way, K_shot=cfg.k_shot, shuffle=True,batch_size=batch_size
        )
    dataloaders["train"] = DataLoader(
        dataset=train_dataset,   num_workers=1,batch_sampler=train_sampler,collate_fn=train_sampler.get_collate_fn()
    )
    val_dataset=AudioDataset('val')
    val_sampler=TaskBatchSampler(
            val_dataset.speaker_id, include_query=True, N_way=cfg.n_way, K_shot=cfg.k_shot, shuffle=True,batch_size=batch_size
        )
    dataloaders["val"] = DataLoader(
        dataset=val_dataset, shuffle=False,
        batch_sampler=val_sampler,collate_fn=val_sampler.get_collate_fn()
    )
    test_dataset=AudioDataset('test')
    # test_sampler=TaskBatchSampler(
    #         test_dataset.speaker_id, include_query=True, N_way=cfg.n_way, K_shot=cfg.k_shot, shuffle=True,batch_size=batch_size
    #     )
    dataloaders["test"] = DataLoader(
        dataset=test_dataset,num_workers=1,
        collate_fn=collate_fn
    )
    return dataloaders