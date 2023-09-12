import argparse
import numpy as np
import torch
import numpy as np
import random
import config as config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models.model_SD import Speaker_model
from cvtransforms import *
import os

from dataloader.dataset_op import XLWBDataModule_ctr as XLWBDataModule_ctr_all

seed = config.random_seed

# set random seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    seed_everything(config.random_seed, workers=True)
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    channel='GREY'
    model = Speaker_model()
    # Transforms 
    train_transforms = Compose([RandomCrop([88, 88]), HorizontalFlip()])
    val_transforms = Compose([CenterCrop([88, 88])])
    test_transforms = Compose([CenterCrop([88, 88])])
    # train
    logger = TensorBoardLogger(
        "SD_logs", name="crop_flip_cl_"+str(config.random_seed), version=config.verison)
    from pytorch_lightning.plugins.training_type import DDPPlugin
    model_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='checkpoints-{epoch:02d}-{val_loss:.2f}'
    )
    trainer = Trainer(strategy=DDPPlugin(find_unused_parameters=True),  # 'ddp',
                      gpus=config.gpus,
                      logger=logger, callbacks=model_checkpoint_callback,
                      precision=config.precision,
                      max_epochs=config.max_epoch,
                      resume_from_checkpoint=config.resume_path
                      )
    XLWBdataset = XLWBDataModule_ctr_all(train_transforms=train_transforms,
                                 val_transforms=val_transforms, test_transforms=test_transforms, channel=channel)
    trainer.fit(model, XLWBdataset)

    # test

    result = trainer.test(model, XLWBdataset, ckpt_path='best')
    
    print(result)
    #record config
    save_path= os.path.join("SD_logs","crop_flip_cl_"+str(config.random_seed),"version_"+str(config.verison),"config.py")
    os.system("rm -f "+save_path)
    os.system("cp config.py "+save_path)


if __name__ == '__main__':
    main()
