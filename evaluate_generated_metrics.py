from utils import Metric
import torch
import numpy as np
import os
from utils import mse_computation
from config import cfg
import pickle
from utils import update_cfg


input_face_paths=[
    '/data3/zhouxukun/MamlFaceBaseline/Imitator/pretrained_model/subj0024_stg02_04seq/voca_eval_with_fixed_test_cond/dump',
    '/data3/zhouxukun/MamlFaceBaseline/Imitator/pretrained_model/subj0138_stg02_04seq/voca_eval_with_fixed_test_cond/dump',

]



def get_upper_msk():
    pkl_path = cfg.path.pkl

    if cfg.dataset == 'vocaset':
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
    elif cfg.dataset == "BIWI":
        # print(" using biwi face")
        # exit(0)
        with open(os.path.join(cfg.path.dataset, "lve.txt")) as f:
            maps = f.read().split(", ")
            mouth_map = [int(i) for i in maps]

        with open(os.path.join(cfg.path.dataset, "fdd.txt")) as f:
            maps = f.read().split(", ")
            upper_map = [int(i) for i in maps]
    return upper_map, mouth_map
if "BIWI" in input_face_paths[0]:
    cfg.dataset='BIWI'
elif 'vocaset' in input_face_paths[0]:
    cfg.dataset='vocaset'
    print("dataset is vocaset ")
update_cfg(cfg)
all_dtw = []
all_lip_l2 = []
all_face_l2 = []
all_lip_max_l2 = []
gt_face_path=f'/data3/zhouxukun/mamlface/{cfg.dataset}/vertices_npy/'
metric=Metric()
all_face_fdd=[]
all_face_lip_max=[]
upper_map, mouth_map=get_upper_msk()
for input_face_path in input_face_paths:
    input_files=os.listdir(input_face_path)

    for file in input_files:
        face=torch.from_numpy(np.load(os.path.join(input_face_path,file),allow_pickle=True))

        if "condition" in file:
            file=file.split('_condition_')[0]+'.npy'
        if not os.path.exists(os.path.join(gt_face_path,file)):
            print(f"file {os.path.join(gt_face_path,file)} not exist")
            continue
        # face=torch.from_numpy(np.load(os.path.join(input_face_path,file),allow_pickle=True))
        target_face=torch.from_numpy(np.load(os.path.join(gt_face_path,file),allow_pickle=True))[::2,:]
        pose_length=face.shape[0]
        pose_length2=target_face.shape[0]
        target_pose=min(pose_length,pose_length2)
        face=face.reshape((pose_length,-1,3))[:target_pose]
        target_face=target_face.reshape((pose_length2,-1,3))[:target_pose]

        motion_std_difference, L2_dis_mouth_max = mse_computation(face, target_face,
                                                                  upper_map,
                                                                  mouth_map)#, vertice_mask)
        lip_dtw=metric.dtw_lip(face,target_face)
        all_dtw.append(lip_dtw.item())
        lip_l2=metric.l2_lip(face,target_face)
        all_lip_l2.append(lip_l2.item())
        face_l2=metric.l2_face(face,target_face)
        all_face_l2.append(face_l2.item())
        lip_max=metric.lip_max_l2(face,target_face)
        all_lip_max_l2.append(lip_max.item())
        all_face_fdd.append(motion_std_difference)
        all_face_lip_max.extend(L2_dis_mouth_max)
print(f"max l2 lip is {np.mean(all_lip_max_l2)}")
print(f"l2 lip is {np.mean(all_lip_l2)}")
print(f"l2 face is {np.mean(all_face_l2)}")
print(f"dtw is {np.mean(all_dtw)}")
print(f"l2_dis_mouth_max is {np.mean(all_lip_max_l2)}")
print(f"Face_fdd is{np.mean(all_face_fdd)}")
print(f"lip_max_fdd is{np.mean(all_face_lip_max)}")

    