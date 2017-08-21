import torch.utils.data
import torchvision.transforms as transforms
from data.base_data_loader import BaseDataLoader
from data.image_folder import ImageFolder
from builtins import object
from pdb import set_trace as st
from data.dataset_mnistedge import *
from data.dataset_mnistedge_unpaired import *


class UnalignedDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)

        train_dataset = MNISTEDGE(root='./datasets/mnistedge',
                                  train=True,
                                  transform=transforms.ToTensor(),
                                  target_transform=transforms.ToTensor())

        train_dataset_unpaired = MNISTEDGE_unpaired(root='./datasets/mnistedge_unpaired',
                                                    train=True,
                                                    transform=transforms.ToTensor(),
                                                    target_transform=transforms.ToTensor())

        test_dataset = MNISTEDGE(root='./datasets/mnistedge',
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  target_transform=transforms.ToTensor())

        self.len_paired = len(train_dataset)
        self.len_unpaired = len(train_dataset_unpaired)
        self.len_test_paired = len(test_dataset)

        #paired

        data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.opt.batchSize,
                                                   shuffle=True)
        data_loader_unpaired = torch.utils.data.DataLoader(dataset=train_dataset_unpaired,
                                                   batch_size=self.opt.batchSize,
                                                   shuffle=True)

        data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset,
                                                           batch_size=self.opt.batchSize,
                                                           shuffle=True)

        #paired
        self.paired_data = data_loader
        #unpaired
        self.unpaired_data = data_loader_unpaired
        #test
        self.test_data = data_loader_test




    def name(self):
        return 'UnalignedDataLoader'

    def load_data_pair(self):
        return self.paired_data, self.len_paired

    def load_data_unpair(self):
        return self.unpaired_data, self.len_unpaired

    def load_data_test(self):
        return self.test_data, self.len_test_paired

    def __len__(self):
        return self.len_paired, self.len_unpaired
