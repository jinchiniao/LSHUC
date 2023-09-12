import os
import sys
sys.path.append(os.getcwd())
from xml.etree.ElementPath import xpath_tokenizer_re
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np
import pytorch_lightning as pl
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import optimizer
import config as config
from dataloader.dataset_pl import Mydataset





class Speaker_model(pl.LightningModule):
    def __init__(self, dropout_p=0.5, classes=500):
        super(Speaker_model, self).__init__()
        # id net
        self.id_net = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(
                1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(
                1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(
                1, 1, 1), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(
                1, 2, 2)),
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(
                1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(96),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(
                1, 2, 2))
        )
        self.id_gru = nn.GRU(
            96*5*5, 1024, 1, bidirectional=True, batch_first=True)

        self.gru1 = nn.GRU(512, 1024, 3, bidirectional=True, batch_first=True, dropout=0.2)



        self._init()
        self.save_hyperparameters()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                

    def forward(self, x):

        # SD feature extractore
        sd = self.id_net(x)
        sd = sd.transpose(1, 2)
        sd = sd.reshape(sd.size(0), sd.size(1), -1)
        _, sd1 = self.id_gru(sd)

        return sd1.transpose(0,1)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=config.base_lr,
                                     weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer, config.base_lr, epochs=config.max_epoch, steps_per_epoch= 1222,pct_start=1.0 / config.max_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, config.base_lr, epochs=config.max_epoch, steps_per_epoch= int(1222*400/config.batch_size/config.gpus),pct_start=0.1)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max = config.max_epoch, eta_min=5e-6)
        return (
            [optimizer],
            [
                {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    'reduce_on_plateau': True,
                    'monitor': 'val_loss',
                }
            ]
        )
        # return ([optimizer],[scheduler])

    def training_step(self, train_batch, batch_idx):
        x1, y1, id1,x2, y2, id2,x3, y3, id3 = train_batch
        assert((id1==id2).all() & (id1!=id3).all())
        #conbine batch

        sd1 = self(x1)
        sd2 = self(x2)
        sd3 = self(x3)
        crit = nn.CrossEntropyLoss()
        #id loss
        crit_ctr = nn.TripletMarginLoss(margin=0.3)
        loss_id = crit_ctr(sd1,sd2,sd3)
        loss = loss_id
        self.log("lr",self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0],on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return {'loss': loss}

    def training_epoch_end(self, outs):
        pass

    def validation_step(self, val_batch, batch_idx):
        x1, y1, id1,x2, y2, id2,x3, y3, id3 = val_batch
        assert((id1==id2).all() & (id1!=id3).all())
        #conbine batch

        sd1 = self(x1)
        sd2 = self(x2)
        sd3 = self(x3)
        crit = nn.CrossEntropyLoss()
        #id loss
        crit_ctr = nn.TripletMarginLoss(margin=0.3)
        loss_id = crit_ctr(sd1,sd2,sd3)
        loss = loss_id
        self.log("val_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return {'loss': loss}

    def validation_epoch_end(self, outs):
        pass

    def test_step(self, test_batch, batch_idx):
        x1, y1, id1,x2, y2, id2,x3, y3, id3 = test_batch
        assert((id1==id2).all() & (id1!=id3).all())
        #conbine batch

        sd1 = self(x1)
        sd2 = self(x2)
        sd3 = self(x3)
        crit = nn.CrossEntropyLoss()
        #id loss
        crit_ctr = nn.TripletMarginLoss(margin=0.3)
        loss_id = crit_ctr(sd1,sd2,sd3)
        loss = loss_id

        self.log("test_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return {'loss': loss}

    def test_step_end(self, outs):
        pass


if __name__ == "__main__":
    import torchsummary
    model = LipNet_SD()
    torchsummary.summary(model, (1, 111, 88, 88), device='cpu')
