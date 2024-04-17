import os

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from args import get_args
from config import cfg
from dataloader.dataset import get_dataloaders
from trainer import MamlTrainer


def main(paraser_args):
    config_file=paraser_args.cfg
    cfg.merge_from_file(config_file)
    if cfg.model.use_pretrained:
        model=MamlTrainer.load_from_checkpoint(cfg.model.pretrained)
    else:
        model = MamlTrainer()

    dataloaders=get_dataloaders(batch_size=cfg.batch_size)

    logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(),'result'), version=1,
        name=f"pretrained{cfg.model.pretrained}_crosstrain{cfg.cross_train}_usetransformer{cfg.model.use_transformer}_backbone{cfg.backbone}_{cfg.lr}lr_{cfg.batch_size}batch_size_{cfg.dataset}dataset_{cfg.feature_dim}feature_dim_{cfg.vertice_dim}vertice_dim{cfg.model.freeze_audio}freeze_audio_")
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{valid_total_loss:.2f}',monitor='valid_total_loss',mode='min',save_top_k=3)
    trainer=pl.Trainer(
        max_epochs=100,logger=logger,log_every_n_steps=10,check_val_every_n_epoch=10,callbacks=[checkpoint_callback],
        accelerator='gpu'
    )
    trainer.fit(model,train_dataloaders=dataloaders['train'],val_dataloaders=dataloaders['val'])
    # trainer.test(model, ckpt_path='last',dataloaders=dataloaders['test'])

if __name__=='__main__':
    args=get_args()
    main(args)

