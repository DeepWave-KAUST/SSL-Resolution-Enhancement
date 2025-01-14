import  torch.utils.data as data
from torch.utils.data.sampler import Sampler
import  os
import torch
import scipy.io as sio
import numpy as np
import math

class Basicdataset(data.Dataset):
    '''
    The items are (datapath).
    Args:
    - dir: the directory where the dataset will be stored
    '''

    def __init__(self, dir):
        self.dir = dir

        self.ids = [os.path.splitext(file)[0] for file in os.listdir(dir) 
                if not file.startswith('.')]

    def __getitem__(self, index):
        idx_file = self.ids[index]

        file = os.path.join(self.dir, idx_file)

        dict = sio.loadmat(file)
        raw = dict['label']

        return {'raw': torch.from_numpy(raw).unsqueeze(0).type(torch.FloatTensor)}

    def __len__(self):
        return len(self.ids)

class EnlargedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch
    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
    """

    def __init__(self, dataset, num_replicas, rank, ratio=1):
        self.dataset = dataset  # dataset used to sample
        self.num_replicas = num_replicas  
        self.rank = rank 
        self.epoch = 0
        # The amount of data selected by each process
        self.num_samples = math.ceil(
            len(self.dataset) * ratio / self.num_replicas)
        # Total data
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # to reduce list memory
        indices = torch.randperm(self.total_size, generator=g)[self.rank:self.total_size:self.num_replicas]
        indices = (indices % len(self.dataset)).tolist()

        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class CUDAPrefetcher():
    """CUDA prefetcher.
    Ref:
    https://github.com/NVIDIA/apex/issues/304#
    It may consums more GPU memory.
    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader, opt=None):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device('cuda')
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return None
        # put tensors to gpu
        with torch.cuda.stream(self.stream):
            if type(self.batch) == dict:
                for k, v in self.batch.items():
                    if torch.is_tensor(v):
                        self.batch[k] = self.batch[k].to(
                            device=self.device, non_blocking=True)
            elif type(self.batch) == list:
                for k in range(len(self.batch)):
                    if torch.is_tensor(self.batch[k]):
                        self.batch[k] = self.batch[k].to(
                            device=self.device, non_blocking=True)
            else:
                assert NotImplementedError

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()
