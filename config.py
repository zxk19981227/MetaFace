import os

from yacs.config import CfgNode as CN

cfg = CN(new_allowed=True)

# 模型参数
cfg.dataset = 'vocaset'
cfg.backbone = 'wav2vec'
cfg.use_pregenerated_feature = False
cfg.audio_fps = 50
cfg.finetune_step=200
cfg.video_fps = 30
cfg.feature_dim = 512
cfg.vertice_dim = 5023 * 3
cfg.min_threshold=0.9
cfg.finetune_lr=1e-4
cfg.lora=False

cfg.device = 'cuda'
cfg.lr = 1e-4
cfg.batch_size = 1
cfg.max_epoch = cfg.batch_size * 100
cfg.cross_train = False
cfg.same_person=False
cfg.save_path='/data3/zhouxukun/visual_talkface_small_path'

cfg.path = CN(new_allowed=True)
cfg.path.project = '/data3/zhouxukun/mamlface'
cfg.path.hubert = '/data3/zhouxukun/transformer_model/hubert-large-ls960-ft'
cfg.path.wav2vec = '/data3/zhouxukun/transformer_model/wav2vec2-large-960h-lv60-self/'
cfg.path.wav2vec_ctc='/data3/zhouxukun/transformer_model/wav2vec2-large-xlsr-53-english'
cfg.path.wav2phonem='/data3/zhouxukun/transformer_model/wav2vec2phoneme/'
cfg.path.phoneme='/data3/zhouxukun/mamlface/vocaset/phonome/fps30'
cfg.path.visualize='/data3/zhouxukun/mamlface/vis/'

cfg.path.save = os.path.join(cfg.path.project, 'result',cfg.dataset)
cfg.path.wav = os.path.join(cfg.path.project, cfg.dataset, 'wav')
cfg.path.save = os.path.join(cfg.path.project, cfg.dataset, 'result','NORMAL')
cfg.path.vertices = os.path.join(cfg.path.project, cfg.dataset, 'vertices_npy')
cfg.path.audio_feature = os.path.join(cfg.path.project, cfg.dataset, cfg.backbone)
cfg.path.template = os.path.join(cfg.path.project, cfg.dataset, 'templates.pkl')
cfg.path.render_template = os.path.join(cfg.path.project, cfg.dataset, 'templates/FLAME_sample.ply')
cfg.path.pkl = os.path.join(cfg.path.project, cfg.dataset, 'FLAME_masks.pkl')
cfg.path.ctc=os.path.join(cfg.path.project, cfg.dataset,'ctc')

cfg.model = CN(new_allowed=True)
cfg.model.freeze_audio = False
cfg.model.use_transformer = False
cfg.model.hidden_dim = 1024
cfg.model.audio_fps = 30
cfg.model.video_fps = 30
cfg.model.personal_dim = 1024
cfg.model.pretrained = None
cfg.model.use_pretrained=False

cfg.n_way=11
cfg.k_shot=1
cfg.lr_inner=1e-4
cfg.num_inner_steps=1

cfg.loss = CN(new_allowed=True)
cfg.loss.l2 = 1000.0  # 正常loss
cfg.loss.freq = 1000.0  # 平滑loss

cfg.train = CN(new_allowed=True)

cfg.train.train_subjects = ("FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA "
                            "FaceTalk_170915_00223_TA  FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA  "
                            "FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
cfg.train.val_subjects = "FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA"
cfg.train.test_subjects = "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA"

cfg.inference = CN(new_allowed=True)
cfg.inference.background_black = True