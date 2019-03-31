

import torch
import torchvision
import os


DEFAULT_ROOT = os.path.abspath('./psvdata')

class PsvDataset(Dataset):
    def __init__(self,  split='all', root=DEFAULT_ROOT):
         self.root = root
         self.split = split

         
