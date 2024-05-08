import os

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from config import cfg




def split_batch(audio, vertices_target, vertices_mask, person_id,template):
    support_audio, query_audio = audio.chunk(2, dim=0)
    support_vertices_gt, query_vertices_gt = vertices_target.chunk(2, dim=0)
    support_vertices_mask, query_vertices_mask = vertices_mask.chunk(2, dim=0)
    support_set_person_id, query_person_id = person_id.chunk(2, dim=0)
    support_template, query_template= template.chunk(2, dim=0)
    return (support_audio,query_audio,support_vertices_gt,query_vertices_gt,support_vertices_mask, query_vertices_mask,
            support_set_person_id,query_person_id,support_template, query_template)


def get_data_path(dataset_type='train', split=None):
    assert dataset_type in ['train', 'val', 'test']
    assert split is not None
    audio_list = os.listdir(cfg.path.wav)
    audio_list.sort()
    data_split = {'train': [i for i in cfg.train.train_subjects.split(" ") if i != ''],
                  'val': [i for i in cfg.train.val_subjects.split(" ") if i != ''],
                  "test": [i for i in cfg.train.test_subjects.split(" ") if i != '']}
    existing_dataset_id = data_split[dataset_type]
    audio_path_list = [
        i for i in audio_list if
                       '_'.join(i.split('_')[:-1]) in existing_dataset_id and int(i.split(".")[0][-2:]) in split[
                           dataset_type]
                       ]
    user_id_dict = {}
    for audio in audio_path_list:
        speaker_name='_'.join(audio.split('_')[:-1])
        if speaker_name not in user_id_dict.keys():
            user_id_dict[speaker_name]=[audio]
        else :
            user_id_dict[speaker_name].append(audio)
    speaker_id_dict={}
    for speaker in data_split[dataset_type]:
        if speaker not in speaker_id_dict.keys():
            speaker_id_dict[speaker] = len(speaker_id_dict.keys())
    return audio_path_list, data_split, speaker_id_dict


def collate_fn(batch):
    audios = []
    vertices = []
    templates = []
    filenames = []
    audio_masks = []
    vertices_masks = []
    speaker_ids = []
    for b in batch:
        audio, vertice, template, speaker_id,filename = b
        vertice_mask = mask_generation(vertice.shape[0])
        audios.append(audio)
        vertices.append(vertice)
        templates.append(template.reshape(-1))
        filenames.append(filename)
        vertices_masks.append(vertice_mask)
        speaker_ids.append(speaker_id)

    audio = pad_sequence(audios, batch_first=True)
    vertice = pad_sequence(vertices, batch_first=True)
    template = torch.stack(templates, dim=0)
    vertices_masks = pad_sequence(vertices_masks, batch_first=True)
    speaker_ids = torch.tensor(speaker_ids)
#
    return (
        audio, vertice, template, filenames,
        vertices_masks, speaker_ids
    )
# Returns a collate function that converts one big tensor into a list of task-specific tensors
# def collate_fn(item_list):
#     # batch componments : audio,vertice,template,speaker_id
#     audio_batch,vertice_batch,template_batch,speaker_batch=[],[],[],[]
#
#     for batch in item_list:
#         audio,vert,temp,speaker=batch
#         audio_batch.append(audio)
#         vertice_batch.append(vert)
#         template_batch.append(temp)
#         speaker_batch.append(speaker)
#     audio_batch=pad_sequence(audio_batch,batch_first=True)
#     vertice_mask_batch=[torch.ones(each.shape) for each in vertice_batch]
#     vertice_batch=pad_sequence(vertice_batch,batch_first=True)
#     vertice_mask_batch=pad_sequence(vertice_mask_batch,batch_first=True)
#     template_batch=pad_sequence(template_batch,batch_first=True)
#     speaker_batch=torch.tensor(speaker_batch)
#
#     return list(zip(audio_batch,vertice_batch,vertice_mask_batch,template_batch,speaker_batch))



def check_path_valid(audio_path, vertices_path):
    vertice_path = os.path.join(vertices_path, audio_path.replace("wav", "npy"))
    if os.path.exists(vertice_path):
        return True
    else:
        return False

3
def mask_generation(length):
    feature_length = torch.ones(length)
    return feature_length


def get_vertice_std(vertice_motion, upper_map):
    L2_dis_upper = np.stack([np.square(vertice_motion[:, v, :]) for v in upper_map], axis=1)
    L2_dis_upper = np.sum(L2_dis_upper, axis=2)
    L2_dis_upper = np.std(L2_dis_upper, axis=0)
    L2_dis_upper = np.mean(L2_dis_upper)
    return L2_dis_upper


def mse_computation(vertice_gt, vertice_pred, upper_map, mouth_map, vertice_mask):
    """

    :param vertice_mask:
    :param mouth_map:
    :param upper_map:
    :param vertice_pred: length,5023*3
    :param vertice_gt: length,5023*3
    :return:
    """
    # face_template = cfg.template
    pred_length = vertice_pred.shape[0]
    gt_length = vertice_gt.shape[0]
    vertice_mask_len = int(torch.sum(vertice_mask).item())
    max_length = min(min(pred_length, gt_length), vertice_mask_len)
    vertices_pred = vertice_pred[:max_length].view(-1, 5023, 3)
    vertices_gt = vertice_gt[:max_length].view(-1, 5023, 3)

    gt_motion_std = get_vertice_std(vertices_gt.cpu().numpy(), upper_map)
    pred_motion_std = get_vertice_std(vertices_pred.cpu().numpy(), upper_map)

    motion_std_difference = gt_motion_std - pred_motion_std
    L2_dis_mouth_max = torch.stack([torch.square(vertices_gt[:, v, :] - vertices_pred[:, v, :]) for v in mouth_map],
                                   dim=1)  #
    L2_dis_mouth_max = torch.sum(L2_dis_mouth_max, dim=2)
    L2_dis_mouth_max = torch.max(L2_dis_mouth_max, dim=1).values
    return motion_std_difference, L2_dis_mouth_max


def linear_interpolation(features, input_fps, output_fps):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


def length_same(gt_vertice, pred_vertice):
    min_le = min(gt_vertice.shape[1], pred_vertice.shape[1])
    gt_vertice = gt_vertice[:, :min_le]
    pred_vertice = pred_vertice[:, :min_le]
    return gt_vertice, pred_vertice


def load_audio(processor, wav_path):
    speech_array, sampling_rate = torchaudio.load(os.path.join(cfg.path.wav, wav_path))

    sampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    speech_array = sampler(speech_array)
    input_values = processor(speech_array, sampling_rate=16000).input_values[0]
    print(input_values)
    return input_values


def load_audio_vertices_pair(audio_path, vertices_path, processor):
    audio_feature = load_audio(processor, audio_path)
    if cfg.dataset == 'vocaset':
        audio_feature = audio_feature[::2, :]
    vertices = np.load(os.path.join(cfg.path.vertices, vertices_path))
    return audio_feature, vertices
