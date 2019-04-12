import torch
import torchvision
import os

from psv import config


class PsvDataset(Dataset):
    def __init__(self,  split='all', root=config.DEFAULT_ROOT):
         self.root = root
         self.split = split


class GsvDataset(Dataset):
    def __init__(self, train=True, fold=1, root=config.DEFAULT_ROOT,
                 size=512,
                 fit='y',
                 crop_to='image',
                 objects_path=None):
        """
        :param train: Whether we generate patches for the training (True) or test (False) set.
        :param fold: Which fold (out of five folds)
        :param root: The root folder of the dataset.
        :param size: The size of each image.  The image is padded or randomly cropped to make it square.
        :param fit: The strategy used to fit an image options are:
                     'y' - fit the entire height of target
                     'xy' - fit the maximum of the width and the height of the target
        :param crop_to: The object to crop to. Can be one of 'image', 'object', or an object label.
        """

        self.root = root
        self.train = train
        self.fold = fold
        if objects_path is None:
            objects_path = f"{self.root}/objects.pkl"
        self.objects_path =  objects_path

        if not os.path.isfile(self.objects_path):
            self._generate_objects_file()

    def _generate_objects_file(self):
        """Generate a per-object dataset"""
        xmls = glob(f'{ROOT}/Annotations/**/*.xml', recursive=True)

    def __len__(self):
        return len(self.objects)
    def __getitem__(self, index:int):
        """
        :param index:
        :return:  A tuple with the input and targets
        """






