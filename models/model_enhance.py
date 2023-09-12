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
from functools import partial


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

    def forward(self, x, enhance,layer_enhance):

        if layer_enhance==0:
            x=x*enhance.repeat(x.shape[0]//enhance.shape[0], 1, 1, 1)
        x = self.layer1(x)

        if layer_enhance==1:
            x=x*enhance.repeat(x.shape[0]//enhance.shape[0], 1, 1, 1)
        x = self.layer2(x)

        if layer_enhance==2:
            x=x*enhance.repeat(x.shape[0]//enhance.shape[0], 1, 1, 1)
        x = self.layer3(x)

        if layer_enhance==3:
            x=x*enhance.repeat(x.shape[0]//enhance.shape[0], 1, 1, 1)
        x = self.layer4(x)

        if layer_enhance==4:
            x=x*enhance.repeat(x.shape[0]//enhance.shape[0], 1, 1, 1)
        x = self.avgpool(x)

        if layer_enhance==5:
            x=x*enhance.repeat(x.shape[0]//enhance.shape[0], 1, 1, 1)
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


class MyDeconvNet(nn.Module):
    def __init__(self,layer=0,mode='e'):
        super(MyDeconvNet, self).__init__()
        self.layer = layer
        self.mode = mode
        if layer==0 or layer==1:
            #[64,22,22]
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=0),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
            self.deconv3 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
            self.deconv4 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.downsample=partial(F.interpolate,size=(22, 22), mode='bilinear', align_corners=False)
        elif layer==2:
            #[128,11,11]
            self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
            self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=0)
        elif layer==3:
            #[256,6,6]
            self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
            self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=1)
        elif layer==4:
            #[512,3,3]
            self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1)
        elif layer==5:
            #[512,1,1]
            self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=1, padding=1)
            self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=1)
        else:
            raise AttributeError
        
        
        self.relu = nn.LeakyReLU(inplace=True)
        if mode=='e':
            self.activation=nn.LeakyReLU(inplace=True)
        elif mode=='d':
            self.activation=nn.Tanh()
        else:
            raise AttributeError
        
    def forward(self, x):
        if self.layer<=2:
            x = self.deconv1(x)
            x = self.deconv2(x)
            x = self.deconv3(x)
            x = self.deconv4(x)
            x = self.downsample(x)
        elif self.layer==3:
            x = self.relu(self.deconv1(x))
            x = self.relu(self.deconv2(x))
            x = self.deconv3(x)
        elif self.layer>3 and self.layer<=5:
            x = self.relu(self.deconv1(x))
            x = self.deconv2(x)
        x = self.activation(x)
        if self.mode=='e':
            x= torch.ones(x.shape,dtype=x.dtype,device=x.device)+torch.abs(x)
        elif self.mode=='d':
            x= torch.ones(x.shape,dtype=x.dtype,device=x.device)-torch.abs(x)
        else:
            raise AttributeError
        return x
class Enhance_model(pl.LightningModule):
    def __init__(self, dropout_p=0.5, classes=500,enhance_layer=0):
        super(Enhance_model, self).__init__()
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
        self.gate_enhance = MyDeconvNet(layer=enhance_layer,mode='e')
        self.enhance_layer=enhance_layer
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
    def sd_forward(self, x):

        # SD feature extractore
        sd = self.id_net(x)
        sd = sd.transpose(1, 2)
        sd = sd.reshape(sd.size(0), sd.size(1), -1)
        _, sd1 = self.id_gru(sd)

        return  sd1.transpose(0,1)      
    def gate_forward(self, x):
        sd = self.id_net(x)
        sd = sd.transpose(1, 2)
        sd = sd.reshape(sd.size(0), sd.size(1), -1)
        _, sd1 = self.id_gru(sd)
        gate_enhance_v=self.gate_enhance(sd1.transpose(0,1).flatten(1,2)[:,:,None,None])
        return sd1.transpose(0,1),gate_enhance_v

    def forward(self, x):

        # SD feature extractore
        sd = self.id_net(x)
        sd = sd.transpose(1, 2)
        sd = sd.reshape(sd.size(0), sd.size(1), -1)
        _, sd1 = self.id_gru(sd)
        gate_enhance_v=self.gate_enhance(sd1.transpose(0,1).flatten(1,2)[:,:,None,None])
        x = self.frontend2plus1D(x)
        # (B, C, T, H, W)->(B, T, C, H, W)
        x = x.transpose(1, 2)
        B, T = x.size(0), x.size(1)
        x = x.contiguous()
        # (B, T, C, H, W)->(B* T, C, H, W)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.resnet18(x,gate_enhance_v,self.enhance_layer)
        # (B,T, 512)->(B,T, 512)
        x = x.view(B, T, -1)
        # To improve the effectiveness of memory(speed up)
        # self.gru1.flatten_parameters()
        x, h = self.gru1(x)


        x = self.FC(self.dropout(x)).mean(1)
        return x,sd1.transpose(0,1),gate_enhance_v

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
                    'monitor': 'train_loss',
                }
            ]
        )
        # return ([optimizer],[scheduler])

    def training_step(self, train_batch, batch_idx):
        x1, y1, id1,x2, y2, id2,x3, y3, id3 = train_batch
        assert((id1==id2).all() & (id1!=id3).all())
        #conbine batch

        y_pred1,sd1,gate1= self(x1)
        sd2,gate2=self.gate_forward(x2)
        sd3,gate3=self.gate_forward(x3)
        # crit = nn.CrossEntropyLoss()
        crit = LabelSmoothingLoss(smoothing=0.1)
        loss_txt = crit(y_pred1, y1)
        wer = (1.0*(y1 != y_pred1.argmax(-1))).mean()
        #id loss
        crit_ctr = nn.TripletMarginLoss(margin=0.3)
        crit_ctr2 = nn.TripletMarginLoss(margin=1)
        
        loss_id = crit_ctr(sd1,sd2,sd3) 
        loss_gate = crit_ctr2(gate1.flatten(1,2).flatten(1,2),gate2.flatten(1,2).flatten(1,2),gate3.flatten(1,2).flatten(1,2))
        loss = loss_txt+config.reg*loss_id + config.reg*loss_gate
        # self.log("lr",self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0],on_step=True,
        #          on_epoch=True, prog_bar=True)
        self.log("train_wer", float(wer), on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log("train_loss_txt", loss_txt, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("train_loss_ctr", loss_id, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("train_loss_gate", loss_gate, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        pred = [Mydataset.words[i] for i in y_pred1.argmax(1)]
        truth_txt = [Mydataset.words[i] for i in y1]
        return {'loss': loss, 'pred': pred, 'truth': truth_txt}

    def training_epoch_end(self, outs):
        predict, truth = outs[0]['pred'], outs[0]['truth']
        for i in range(1):
            print('{:\u3000<25}|{:>25}'.format(predict[i], truth[i]))

    def validation_step(self, val_batch, batch_idx):
        x, y, id = val_batch
        y_pred,_,_= self(x)
        crit = nn.CrossEntropyLoss()
        loss = crit(y_pred, y)

        wer = (1.0*(y != y_pred.argmax(-1))).mean()
        self.log("val_wer", float(wer), on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
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
        y_pred,_,_ = self(x)
        crit = nn.CrossEntropyLoss()
        loss = crit(y_pred, y)
        a=(1.0*(y != y_pred.argmax(-1)))
        wer = (1.0*(y != y_pred.argmax(-1))).mean()
        self.log("test_wer", float(wer), on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
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

