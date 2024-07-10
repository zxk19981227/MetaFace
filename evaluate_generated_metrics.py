from utils import Metric
import torch
import numpy as np
import os
from utils import mse_computation


input_face_paths=['/data3/zhouxukun/MamlFaceBaseline/Imitator/pretrained_model/subj0138_stg02_04seq/voca_eval_with_fixed_test_cond/dump/',
                  '/data3/zhouxukun/MamlFaceBaseline/Imitator/pretrained_model/subj0024_stg02_04seq/voca_eval_with_fixed_test_cond/dump/'

]
all_dtw = []
all_lip_l2 = []
all_face_l2 = []
all_lip_max_l2 = []
gt_face_path='/data3/zhouxukun/mamlface/vocaset/vertices_npy/'
metric=Metric()
for input_face_path in input_face_paths:
    input_files=os.listdir(input_face_path)

    for file in input_files:

        face=torch.from_numpy(np.load(os.path.join(input_face_path,file),allow_pickle=True))
        target_face=torch.from_numpy(np.load(os.path.join(gt_face_path,file),allow_pickle=True))
        pose_length=face.shape[0]
        pose_length2=target_face.shape[0]
        target_pose=min(pose_length,pose_length2)
        face=face.reshape((pose_length,-1,3))[:target_pose]
        target_face=target_face.reshape((pose_length2,-1,3))[:target_pose]

        lip_dtw=metric.dtw_lip(face,target_face)
        all_dtw.append(lip_dtw.item())
        lip_l2=metric.l2_lip(face,target_face)
        all_lip_l2.append(lip_l2.item())
        face_l2=metric.l2_face(face,target_face)
        all_face_l2.append(face_l2.item())
        lip_max=metric.lip_max_l2(face,target_face)
        all_lip_max_l2.append(lip_max.item())
print(f"max l2 lip is {np.mean(all_lip_max_l2)}")
print(f"l2 lip is {np.mean(all_lip_l2)}")
print(f"l2 face is {np.mean(all_face_l2)}")
print(f"dtw is {np.mean(all_dtw)}")

    