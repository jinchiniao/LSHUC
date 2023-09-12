import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset
from turbojpeg import TurboJPEG, TJPF_BGR,TJPF_GRAY
from cvtransforms import *
import config as config
import editdistance



class Mydataset(Dataset):
    """
    Load LRW dataset
    """
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ']
    word_dict = 'label_sorted.txt'
    words = []
    with open(word_dict) as f:
        lines = f.readlines()
        for line in lines:
            words.append(line.replace("\n", ""))

    def __init__(self, path, split, cvtransforms=None,channel='GREY',insert=False):
        super().__init__()
        self.path = path
        self.split = split
        self.cvtransforms = cvtransforms
        self.channel=channel
        self.insert=insert
        # decode the video from jpg string to jpg
        self.decoder = TurboJPEG()
        self.data = []
        self.id = []

        self.__load_data__()

    def __getitem__(self, index):
        video_encode = []
        dat=torch.load(self.data[index])
        try:
            video_encode = dat.get('video')
            if len(video_encode) == 0:
                print(self.data[index], "error")
        except:
            print(self.data[index], "error")
        frames = []
        for i in range(len(video_encode)):
            if self.channel=='GREY':
                image = self.decoder.decode(video_encode[i],pixel_format=TJPF_GRAY)
            elif self.channel == 'RGB':
                image = self.decoder.decode(video_encode[i],pixel_format=TJPF_BGR)
            else:
                print('decode channel error.')
            if i!=0 and self.insert==True:
                frames.append((image+last_frame)/2)
            frames.append(image)
            last_frame=image
        feat = np.stack(frames).transpose(3, 0, 1, 2) / 255.0
        if self.cvtransforms:
            feat = self.cvtransforms(feat)
        label_idx = dat.get('label')
        id_idx = self.id[index]
        # add <sos> and <eos>
        return torch.FloatTensor(np.ascontiguousarray(feat)), label_idx, int(id_idx)

    def __len__(self):
        return len(self.data)

    def __load_data__(self):
        # transcript
        idx = 0
        split_path=config.split_path+"LRW_ID_"+self.split+".txt"
        with open(split_path) as myfile:
            samples = myfile.read().splitlines()      
        for (i, sample) in enumerate(samples):
            label,file=sample.split(' ')    
            data=os.path.join(self.path,file)+".pkl"       
            self.data.append(data) 
            self.id.append(label) 
        print("load {} in total finally".format(len(self.data)))

    @staticmethod
    def txt2arr(txt, start=1):
        arr = []
        for c in list(txt):
            arr.append(Mydataset.letters.index(c) + start)
        return np.array(arr)

    @staticmethod
    def arr2txt(arr, start=1):
        txt = []
        for n in arr:
            if n == len(Mydataset.letters):
                break
            if(n >= start):
                txt.append(Mydataset.letters[n - start])
        return ''.join(txt).strip()

    @staticmethod
    @staticmethod
    def at_arr2txt(arr):
        txt = []
        for n in arr:
            if n == len(Mydataset.letters) or n == 1:
                break
            elif int(n) not in Mydataset.letters:
                txt.append('<UNK>')
            else:
                txt.append(Mydataset.letters[int(n)])
        return ''.join(txt).strip()

    @staticmethod
    # use ctc rules to squeeze the array
    def ctc_arr2txt(arr, start=1):
        pre = -1
        txt = []
        for n in arr:
            if n == len(Mydataset.letters):
                break
            if(pre != n and n >= start):
                if(len(txt) > 0 and txt[-1] == ' ' and Mydataset.letters[n - start] == ' '):
                    pass
                else:
                    txt.append(Mydataset.letters[n - start])
            pre = n
        return ''.join(txt).strip()

    @staticmethod
    def wer(predict, truth):
        word_pairs = [(p[0].split(' '), p[1].split(' '))
                      for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return wer

    @staticmethod
    def cer(predict, truth):
        cer = [1.0*editdistance.eval(p[0], p[1])/(1 if len(p[1]) == 0 else len(p[1]))
               for p in zip(predict, truth)]
        return cer


class XLWBDataModule(pl.LightningDataModule):
    def __init__(self, train_transforms, val_transforms, test_transforms,channel='GREY',insert=False):
        super().__init__()
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.is_distributed = config.gpus > 1
        self.channel = channel
        self.insert = insert

    def train_dataloader(self):
        train_split = Mydataset(
            config.path,'train',  cvtransforms=self.train_transforms, channel=self.channel,insert=self.insert)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_split,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            shuffle=True
        )
        return train_loader

    def val_dataloader(self):
        val_split = Mydataset(
            config.path, 'val',  cvtransforms=self.test_transforms, channel=self.channel,insert=self.insert)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_split,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            shuffle=False
        )
        return val_loader

    def test_dataloader(self):
        test_split = Mydataset(
            config.path, 'test',  cvtransforms=self.test_transforms, channel=self.channel,insert=self.insert)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_split,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            shuffle=False
        )
        return test_loader


if __name__ == "__main__":
    from tqdm import tqdm
    train_transforms = Compose([
        RandomCrop((88, 88)),HorizontalFlip(0)
    ])
    XLWBdataset = XLWBDataModule(train_transforms=train_transforms,
                                 val_transforms=train_transforms, test_transforms=train_transforms)

    for data in tqdm(XLWBdataset.test_dataloader()):
        pass

