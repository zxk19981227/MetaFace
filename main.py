import os

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from args import get_args
from config import cfg
from dataloader.dataset import get_dataloaders
from trainer import MamlTrainer


def main(paraser_args):
    config_file = paraser_args.cfg
    cfg.merge_from_file(config_file)
    if cfg.model.use_pretrained:
        model = MamlTrainer.load_from_checkpoint(cfg.model.pretrained)
    else:
        model = MamlTrainer()

    dataloaders = get_dataloaders(batch_size=cfg.batch_size)
    default_root_path=f"pretrained{cfg.model.pretrained}_crosstrain{cfg.cross_train}_usetransformer{cfg.model.use_transformer}_backbone{cfg.backbone}_{cfg.lr}lr_{cfg.batch_size}batch_size_{cfg.dataset}dataset_{cfg.feature_dim}feature_dim_{cfg.vertice_dim}vertice_dim{cfg.model.freeze_audio}freeze_audio_"

    logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), 'result'), version=1,
        name=default_root_path)
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{valid_total_loss:.10f}',monitor='valid_total_loss', mode='min', save_top_k=3)
    trainer = pl.Trainer(
        max_epochs=1000, logger=logger , log_every_n_steps=1, check_val_every_n_epoch=1, callbacks=[checkpoint_callback],)
    trainer.fit(model, train_dataloaders=dataloaders['train'], val_dataloaders=dataloaders['val'])


if __name__ == '__main__':
    args = get_args()
    main(args)
