import argparse
import numpy as np
import torch
import numpy as np
import random
import config as config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models.model_enhance import Enhance_model

from cvtransforms import *

import os


from dataloader.dataset_op import XLWBDataModule as XLWBDataModule_ctr

import gc

seed = config.random_seed

# set random seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(args):
    seed_everything(config.random_seed, workers=True)
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    if not(args.SD_model_path and args.baseline_model_path):
        print("please input SD_model_path and baseline_model_path")
        return
    channel='GREY'
    model = Enhance_model()
    cvtransforms = True
    model_dict = model.state_dict() 
    if args.SD_model_path:
        pretrained_dict_id = torch.load(args.SD_model_path,map_location='cpu')[
            'state_dict']
        pretrained_dict_id = {k: v for k,
                            v in pretrained_dict_id.items() if k in model_dict}
        model_dict.update(pretrained_dict_id)
    if args.baseline_model_path:
        pretrained_dict_txt = torch.load(args.baseline_model_path,map_location='cpu')[
            'state_dict']
        pretrained_dict_txt = {k: v for k,
                    v in pretrained_dict_txt.items() if k in model_dict}
        model_dict.update(pretrained_dict_txt)
    model.load_state_dict(model_dict)
    gc.collect()
    # Transforms 
    train_transforms = Compose([RandomCrop([88, 88]), HorizontalFlip()])
    val_transforms = Compose([CenterCrop([88, 88])])
    test_transforms = Compose([CenterCrop([88, 88])])
    # train
    logger = TensorBoardLogger(
        "Enhance_logs", name="crop_flip_cl_"+str(config.random_seed), version=config.verison)
    from pytorch_lightning.plugins.training_type import DDPPlugin
    model_checkpoint_callback = ModelCheckpoint(
        monitor='val_wer',
        filename='checkpoints-{epoch:02d}-{val_loss:.2f}-{val_wer:.2f}'
    )
    trainer = Trainer(strategy=DDPPlugin(find_unused_parameters=True),  # 'ddp',
                      gpus=config.gpus,
                      logger=logger, callbacks=model_checkpoint_callback,
                      precision=config.precision,
                      max_epochs=config.max_epoch,
                      resume_from_checkpoint=config.resume_path
                      )
    XLWBdataset = XLWBDataModule_ctr(train_transforms=train_transforms,
                                 val_transforms=val_transforms, test_transforms=test_transforms, channel=channel)
    trainer.fit(model, XLWBdataset)

    # test

    result = trainer.test(model, XLWBdataset, ckpt_path='best')
    
    print(result)
    #record config
    save_path= os.path.join("Enhance_logs","crop_flip_cl_"+str(config.random_seed),"version_"+str(config.verison),"config.py")
    os.system("rm -f "+save_path)
    os.system("cp config.py "+save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Learning Separable Hidden Unit Contributions for Speaker-Adaptive Lip-Reading')
    parser.add_argument(
        '--SD_model_path', type=str, default=None, help='model')
    parser.add_argument(
        '--baseline_model_path', type=str, default=None, help='model')

    args = parser.parse_args()

    main(args)
