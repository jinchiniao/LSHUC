import argparse
import numpy as np
import torch
import numpy as np
import random
import config as config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models.model_ensemble import Ensemble_model
from cvtransforms import *
import os


from dataloader.dataset_op import XLWBDataModule as XLWBDataModule_ctr

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
    channel='GREY'
    model=Ensemble_model()
    #load pretrain model
    model_dict = model.state_dict() 
    pretrained_dict = torch.load(args.enhance_model_path,map_location='cpu')[
        'state_dict']
    pretrained_dict = {k: v for k,
                        v in pretrained_dict.items() if k in model_dict}
    print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    logger = TensorBoardLogger(
        "Ensemble_logs", name="crop_flip_cl"+str(config.random_seed), version=config.verison)
    train_transforms = Compose([RandomCrop([88, 88]), HorizontalFlip()])
    val_transforms = Compose([CenterCrop([88, 88])])
    test_transforms = Compose([CenterCrop([88, 88])])
    model_checkpoint_callback = ModelCheckpoint(
        monitor='val_wer',
        filename='checkpoints-{epoch:02d}-{val_loss:.2f}-{val_wer:.4f}'
    )
    from pytorch_lightning.plugins.training_type import DDPPlugin
    trainer = Trainer(strategy=DDPPlugin(find_unused_parameters=True),  # 'ddp',
                      gpus=config.gpus,
                      logger=logger, callbacks=model_checkpoint_callback,
                      precision=config.precision,
                      max_epochs=config.max_epoch,
                      resume_from_checkpoint=None
                      )
    XLWBdataset = XLWBDataModule_ctr(train_transforms=train_transforms,
                                val_transforms=val_transforms, test_transforms=test_transforms, channel=channel)
    trainer.fit(model, XLWBdataset)
    # test

    result = trainer.test(model, XLWBdataset, ckpt_path='best')
    
    print(result)
    #record config
    save_path= os.path.join("Ensemble_logs","crop_flip_cl"+str(config.random_seed),"version_"+str(config.verison),"config.py")
    os.system("rm -f "+save_path)
    os.system("cp config.py "+save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Learning Separable Hidden Unit Contributions for Speaker-Adaptive Lip-Reading')
    parser.add_argument(
        '--enhance_model_path', type=str, default=None, help='model')

    args = parser.parse_args()

    main(args)