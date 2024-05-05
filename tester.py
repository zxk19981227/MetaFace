from args import get_args
from dataloader.dataset import get_dataloaders
from trainer import MamlTrainer
from config import cfg
import lightning as pl
def test_file(paraser_args):
    config_file = paraser_args.cfg
    cfg.merge_from_file(config_file)
    dataloader=get_dataloaders(batch_size=1)
    test_dataloader=dataloader['test']
    trainer=pl.Trainer()
    model=MamlTrainer.load_from_checkpoint('/data3/zhouxukun/mamlface/result/pretrainedNone_crosstrainFalse_usetransformerFalse_backbonehubert_0.0001lr_1batch_size_vocasetdataset_512feature_dim_15069vertice_dimFalsefreeze_audio_/version_1/checkpoints/epoch=929-valid_total_loss=0.0003739640.ckpt')
    trainer.test(model,dataloaders=test_dataloader)

if __name__ == '__main__':
    args = get_args()
    test_file(args)

