import os

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MUJOCO_GL"] = "osmesa"
from render_utils import render_sequence_meshes
from psbody.mesh import Mesh
import numpy as np
from utils import update_cfg
from config import cfg


def render_npy_files(npy_file_name,template,save_path,npy_path):
    vertices=np.load(os.path.join(npy_path,npy_file_name))
    if "_condition_" in npy_file_name:
        npy_file_name = npy_file_name.split("_condition_")[0] + '.npy'
    render_sequence_meshes(
    audio_path=os.path.join(cfg.path.wav,npy_file_name.replace('.npy','.wav')),sequence_vertices=vertices,
    template=template, out_path=save_path,
    file_type=npy_file_name.replace('.mp4','') + '_pred', ft=None,
    vt=None, tex_img=None,
    )

def main():
    from config import cfg
    numpy_safe_paths=[#"/data3/zhouxukun/MamlFaceBaseline/selftalk_finetune_voca/",
        # "/data3/zhouxukun/MamlFaceBaseline/FaceFormer_finetune/BIWI/result/"
        '/data3/zhouxukun/MamlFaceBaseline/voca-biwi/biwi_result',
        #  "/data3/zhouxukun/MamlFaceBaseline/SelfTalk_release/BIWI/result_1024_07_22_10_42",
        # "/data3/zhouxukun/MamlFaceBaseline/TalkingStyle/BIWI/save_best/result_b"
    #
    ]
        #"/data3/zhouxukun/MamlFaceBaseline/Imitator/pretrained_model/subj0024_stg02_04seq/voca_eval_with_fixed_test_cond/dump/",  "/data3/zhouxukun/MamlFaceBaseline/Imitator/pretrained_model/subj0138_stg02_04seq/voca_eval_with_fixed_test_cond/dump/"]
    #"/data3/zhouxukun/MamlFaceBaseline/vocaset_output"
    #"/data3/zhouxukun/MamlFaceBaseline/TalkingStyle/vocaset/save_best/result_b"
    #'/data3/zhouxukun/MamlFaceBaseline/FaceFormer_finetune/vocaset/result'
    for numpy_safe_path in numpy_safe_paths:
        # if 'BIWI' in numpy_safe_path or 'biwi' in numpy_safe_path:
        #     cfg.datasaet='BIWI'
        # else:
        #     cfg.dataset='BIWI'
        cfg.dataset = 'BIWI'
        cfg.video_fps=25
        cfg=update_cfg(cfg)
        print(f"cfg.dataset is {cfg.dataset}")
        print(numpy_safe_path)
        template_mesh = Mesh(filename=cfg.path.render_template)

        npy_lists=os.listdir(numpy_safe_path)
        save_path=numpy_safe_path+'_visualize'
        for file in npy_lists:
            if not file.endswith('.npy'):
                print(file)
                continue
            render_npy_files(file,template_mesh,save_path,numpy_safe_path)

if __name__=="__main__":
    main()

    

