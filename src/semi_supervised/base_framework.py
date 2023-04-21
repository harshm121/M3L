import torch

from abc import ABC, abstractmethod

class AbstractFramework(ABC):
    def __init__(self, seg_model, args):
        self.rank = args.rank
        if self.rank == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        self.eval_flip = args.eval_flip
        self.seg_model = seg_model
        self.snapshot_dir = args.snapshot_dir
        self.snapshot_iter = args.snapshot_iter
        self.snapshot_epoch = args.snapshot_epoch
        self.world_size = args.world_size
        self.sliding_eval = args.sliding_eval
        
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def evaluate(self):
        pass