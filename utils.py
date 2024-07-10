import os

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from config import cfg
from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf
import pickle
from sklearn.metrics.pairwise import euclidean_distances
def update_cfg(cfg):
    cfg.path.save = os.path.join(cfg.path.project, 'result', cfg.dataset)
    cfg.path.wav = os.path.join(cfg.path.project, cfg.dataset, 'wav')
    cfg.path.save = os.path.join(cfg.path.project, cfg.dataset, 'result', 'NORMAL')
    cfg.path.vertices = os.path.join(cfg.path.project, cfg.dataset, 'vertices_npy')
    cfg.path.audio_feature = os.path.join(cfg.path.project, cfg.dataset, cfg.backbone)
    cfg.path.template = os.path.join(cfg.path.project, cfg.dataset, 'templates.pkl')
    cfg.path.render_template = os.path.join(cfg.path.project, cfg.dataset, 'FLAME_sample.ply')
    cfg.path.pkl = os.path.join(cfg.path.project, cfg.dataset, 'FLAME_masks.pkl')
    cfg.path.ctc = os.path.join(cfg.path.project, cfg.dataset, 'ctc')
    cfg.path.dataset = os.path.join(cfg.path.project, cfg.dataset)
    return cfg

def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):

                D1[i, j] = torch.mean(dist(x[i], y[j]))
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1]/len(path[0]), C, D1, path


def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)


if __name__ == '__main__':
    w = inf
    s = 1.0
    if 1:  # 1-D numeric
        from sklearn.metrics.pairwise import manhattan_distances
        x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
        y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
        dist_fun = manhattan_distances
        w = 1
        # s = 1.2
    elif 0:  # 2-D numeric
        from sklearn.metrics.pairwise import euclidean_distances
        x = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [4, 3], [2, 3], [1, 1], [2, 2], [0, 1]]
        y = [[1, 0], [1, 1], [1, 1], [2, 1], [4, 3], [4, 3], [2, 3], [3, 1], [1, 2], [1, 0]]
        dist_fun = euclidean_distances

    dist, cost, acc, path = dtw(x, y, dist_fun, w=w, s=s)

    # Vizualize
    from matplotlib import pyplot as plt
    plt.imshow(cost.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
    plt.plot(path[0], path[1], '-o')  # relation
    plt.xticks(range(len(x)), x)
    plt.yticks(range(len(y)), y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('tight')
    if isinf(w):
        plt.title('Minimum distance: {}, slope weight: {}'.format(dist, s))
    else:
        plt.title('Minimum distance: {}, window widht: {}, slope weight: {}'.format(dist, w, s))
    plt.show()

def distance_face(x,y):
    x=x.view(-1,3)
    y=y.view(-1,3)
    distance_x_y=torch.sqrt(torch.sum((x-y)**2,dim=-1))
    distance_x_y=torch.mean(distance_x_y)
    return distance_x_y
class Metric:
    def __init__(self):
        self.l2_function=F.mse_loss
        with open('pkls/FLAME_masks.pkl','rb') as f:
            self.lip_mask = pickle.load(f,encoding='latin1')["lips"]
            
    # def lip_sync(self,input_mesh_sequence,target_mesh_sequence):
    #     """
    #     shape for sequence: (seq_len,5023,3)
    #     """
    #     assert input_mesh_sequence.shape==target_mesh_sequence
    #     mse_loss=self.l2_function(input_mesh_sequence,target_mesh_sequence,reduction='none')
    #     lip_sync_loss=torch.sum(mse_loss,dim=2)
    #     lip_sync_loss=torch.max(lip_sync_loss,dim=1)
    #     return lip_sync_loss
    def l2_face(self,lip_predict,lip_real):
        # l2_face=self.l2_function(input_mesh_sequence,target_mesh_sequence,reduction='none')
        seq_len=lip_predict.shape[0]
        pred_verts_mm = lip_predict.view(seq_len, -1, 3) * 1000.0
        gt_verts_mm = lip_real.view(seq_len, -1, 3) * 1000.0

        diff_in_mm = pred_verts_mm - gt_verts_mm
        l2_dist_in_mm = torch.sqrt(torch.sum(diff_in_mm ** 2, dim=-1))
        # max_l2_error_lip_vert, idx = torch.mean(l2_dist_in_mm, dim=-1)
        mean_max_l2_error_face_vert = torch.mean(l2_dist_in_mm )
        return  mean_max_l2_error_face_vert
    def dtw_lip(self,input_mesh_sequence,target_mesh_sequence):
        pose_length=input_mesh_sequence.shape[0]
        # print(f"shape for lip is {input_mesh_sequence.shape}")
        # print(self.lip_mask)
        # print(f"shape for input_mesh_sequence is {input_mesh_sequence.shape}")
        input_mesh_sequence=input_mesh_sequence[:,self.lip_mask]

        # print(f"shape for input_mesh_sequence is {input_mesh_sequence.shape}")
        target_mesh_sequence=target_mesh_sequence[:,self.lip_mask]
        input_mesh_sequence=input_mesh_sequence*1000
        target_mesh_sequence=target_mesh_sequence*1000
        print(input_mesh_sequence.shape)
        dtw_loss,_,_,_=dtw(
            input_mesh_sequence.view(pose_length,-1,3).cpu(),target_mesh_sequence.view(pose_length,-1 ,3).cpu(),
            dist=distance_face
                           )
        return  dtw_loss
    def lip_max_l2(self, lip_predict, lip_real):
        """
        This is the lip sync metric used in the faceformer paper
        """
        # mask = mask.to(real.device)
        # lip_pred = predict * mask
        # lip_real = real * mask
        seq_len=lip_predict.shape[0]
        pred_verts_mm = lip_predict.view(seq_len, -1, 3) * 1000.0
        gt_verts_mm = lip_real.view(seq_len, -1, 3) * 1000.0

        diff_in_mm = pred_verts_mm - gt_verts_mm
        l2_dist_in_mm = torch.sqrt(torch.sum(diff_in_mm ** 2, dim=-1))
        max_l2_error_lip_vert, idx = torch.max(l2_dist_in_mm, dim=-1)
        mean_max_l2_error_lip_vert = torch.mean(max_l2_error_lip_vert)
        return mean_max_l2_error_lip_vert
    def l2_lip(self,lip_predict,lip_real):
        seq_len=lip_predict.shape[0]
        pred_verts_mm = lip_predict.view(seq_len, -1, 3) * 1000.0
        gt_verts_mm = lip_real.view(seq_len, -1, 3) * 1000.0

        diff_in_mm = pred_verts_mm - gt_verts_mm
        l2_dist_in_mm = torch.sqrt(torch.sum(diff_in_mm ** 2, dim=-1))[:,self.lip_mask]
        # max_l2_error_lip_vert, idx = torch.mean(l2_dist_in_mm, dim=-1)
        mean_max_l2_error_lip_vert = torch.mean(l2_dist_in_mm)
        return mean_max_l2_error_lip_vert
        
        



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
    print(split)
    audio_path_list = [
        i for i in audio_list if
                       '_'.join(i.split('_')[:-1]) in existing_dataset_id and int(i.split(".")[0][-2:]) in split
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




def check_path_valid(audio_path, vertices_path):
    vertice_path = os.path.join(vertices_path, audio_path.replace("wav", "npy"))
    if os.path.exists(vertice_path):
        return True
    else:
        return False


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
    vertices_pred = vertice_pred[:max_length].view(max_length, -1, 3)
    vertices_gt = vertice_gt[:max_length].view(max_length, -1, 3)

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
