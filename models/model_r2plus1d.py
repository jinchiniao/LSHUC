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



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se
        
        if(self.se):
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.conv3 = conv1x1(planes, planes//16)
            self.conv4 = conv1x1(planes//16, planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        if(self.se):
            w = self.gap(out)
            w = self.conv3(w)
            w = self.relu(w)
            w = self.conv4(w).sigmoid()
            
            out = out * w
        
        out = out + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, se=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.se = se
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.bn = nn.BatchNorm1d(512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        return x        

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, input, target):
        target = target.unsqueeze(1)
        target_one_hot = torch.zeros_like(input)
        target_one_hot.scatter_(1, target, 1)

        smoothed_target = target_one_hot * self.confidence + (1 - target_one_hot) * self.smoothing / (input.size(1) - 1)

        log_prob = nn.functional.log_softmax(input, dim=1)
        loss = nn.functional.kl_div(log_prob, smoothed_target, reduction='batchmean')

        return loss

class R2plus1d(pl.LightningModule):
    def __init__(self, dropout_p=0.5, classes=500):
        super(R2plus1d, self).__init__()
        # frontend3D
        self.frontend2plus1D = nn.Sequential(
            nn.Conv3d(1, 45, kernel_size=(1, 7, 7), stride=(
                1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1), stride=(
                1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(
                1, 2, 2), padding=(0, 1, 1))
        )

        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])

        self.gru1 = nn.GRU(512, 1024, 3, bidirectional=True, batch_first=True, dropout=0.2)


        self.FC = nn.Linear(1024*2, classes)
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
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

        x = self.frontend2plus1D(x)
        # (B, C, T, H, W)->(B, T, C, H, W)
        x = x.transpose(1, 2)
        B, T = x.size(0), x.size(1)
        x = x.contiguous()
        # (B, T, C, H, W)->(B* T, C, H, W)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.resnet18(x)
        # (B,T, 512)->(B,T, 512)
        x = x.view(B, T, -1)


        x, h = self.gru1(x)


        x = self.FC(self.dropout(x)).mean(1)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=config.base_lr,
                                     weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, config.base_lr, epochs=config.max_epoch, steps_per_epoch= int(1222*400/config.batch_size/config.gpus),pct_start=0.1)
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

    def training_step(self, train_batch, batch_idx):
        x, y, id = train_batch
        y_pred = self(x)
        crit = LabelSmoothingLoss(smoothing=0.1)
        loss = crit(y_pred, y)
        wer = (1.0*(y != y_pred.argmax(-1))).mean()
        self.log("lr",self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0],on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log("train_wer", float(wer), on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        pred = [Mydataset.words[i] for i in y_pred.argmax(1)]
        truth_txt = [Mydataset.words[i] for i in y]
        return {'loss': loss, 'pred': pred, 'truth': truth_txt}

    def training_epoch_end(self, outs):
        predict, truth = outs[0]['pred'], outs[0]['truth']
        for i in range(1):
            print('{:\u3000<25}|{:>25}'.format(predict[i], truth[i]))

    def validation_step(self, val_batch, batch_idx):
        x, y, id = val_batch
        y_pred = self(x)
        crit = nn.CrossEntropyLoss()

        loss = crit(y_pred, y)

        wer = (1.0*(y != y_pred.argmax(-1))).mean()
        self.log("val_wer", float(wer), on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        pred = [Mydataset.words[i] for i in y_pred.argmax(1)]
        truth_txt = [Mydataset.words[i] for i in y]
        return {'loss': loss, 'pred': pred, 'truth': truth_txt}

    def validation_epoch_end(self, outs):
        predict, truth = outs[0]['pred'], outs[0]['truth']
        for i in range(1):
            print('{:\u3000<25}|{:>25}'.format(predict[i], truth[i]))

    def test_step(self, test_batch, batch_idx):
        x, y, id = test_batch
        y_pred = self(x)
        crit = nn.CrossEntropyLoss()
        loss = crit(y_pred, y)
        a=(1.0*(y != y_pred.argmax(-1)))
        wer = (1.0*(y != y_pred.argmax(-1))).mean()
        self.log("test_wer", float(wer), on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log("test_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        pred = [Mydataset.words[i] for i in y_pred.argmax(1)]
        truth_txt = [Mydataset.words[i] for i in y]
        return {'loss': loss, 'pred': pred, 'truth': truth_txt}

    def test_step_end(self, outs):
        predict, truth, = outs['pred'], outs['truth']
        length = 3 if len(outs['pred']) > 3 else len(outs['pred'])
        for i in range(length):
            print('{:\u3000<25}|{:>25}'.format(predict[i], truth[i]))


if __name__ == "__main__":
    import torchsummary
    model = R2plus1d()
    torchsummary.summary(model, (3, 111, 96, 96), device='cpu')
