"""PyTorch Dataset(s) for the PSV data. 


"""
import os
from glob import glob

from psv import Annotation
from PIL import Image  # , ImageDraw

from psv.config import C
from torch.utils.data import Dataset

from psv import transforms


class PsvDataset(Dataset):
    def __init__(self,
                 split=None,
                 root=C.DATA_ROOT,
                 transform=None,
                 download=True,
                 url=C.DATA_URL,
                 pwd: bytes = None
                 ):
        """
        :param split: A split of the data (see the txt files under DATA_ROOT).
        :param root: The root folder of the dataset.
        """

        self.root = root

        if download == 'force':
            _download = True
        elif download and not os.path.isdir(os.path.join(self.root, 'Images')):
            _download = True
        else:
            _download = False
        if _download:
            download_images(root=self.root, pwd=pwd)

        if split is None:
            xmls = sorted(glob(f'{self.root}/Annotations/**/*.xml', recursive=True))
        else:
            xmls = [line.strip() for line in open(f'{self.root}/{split}.txt')]
        self.xmls = xmls

        if transform is None:
            transform = transforms.ToSegmentation()
        self.transform = transform

    def get_annotation(self, index):
        return Annotation(self.xmls[index])

    def __len__(self):
        return len(self.xmls)

    def __getitem__(self, index: int):
        """
        :param index:
        :return:  A tuple with the input and targets
        """
        a = self.get_annotation(index)
        im = a.image

        result = self.transform(image=im, annotation=a)

        return result


def download_images(url=None, root=None, zipname='gsv-images.zip', pwd: bytes = None):
    if url is None:
        url = C.DATA_URL
    if root is None:
        root = C.DATA_ROOT

    import wget
    wget.download(url, zipname)
    extract_images(root, zipname, pwd)


def extract_images(root=None, zipname=None, pwd: bytes = None):
    import zipfile
    import tqdm as tq

    if root is None:
        root = C.DATA_ROOT

    if zipname is None:
        zip_path = os.path.join(C.DATA_ROOT, 'gsv-images.zip')
    else:
        zip_path = zipname

    with zipfile.ZipFile(zip_path) as zf:
        # Loop over each file
        for file in tq.tqdm(zf.namelist(), total=len(zf.namelist())):
            zf.extract(member=file, path=root, pwd=pwd)


TFM_SEGMENTATION = transforms.Compose(
    transforms.ToSegmentation(),
    transforms.DropKey('annotation'),
    transforms.ToTensor('image'),
    transforms.ToTensor('mask', preserve_range=True),
)

TFM_SEGMENTATION_CROPPED = transforms.Compose(
    transforms.ToSegmentation(),

    # Crop in on the facades
    transforms.SetCropToFacades(pad=20, pad_units='percent', skip_unlabeled=True, minsize=(512, 512)),
    transforms.ApplyCrop('image'),
    transforms.ApplyCrop('mask'),

    # Resize the height to fit in the net (with some wiggle room)
    transforms.Resize('image', height=600),
    transforms.Resize('mask', height=600, interpolation=Image.NEAREST),

    # Randomly choose a subimage
    transforms.SetRandomCrop(512, 512),
    transforms.ApplyCrop('image'),
    transforms.ApplyCrop('mask'),

    transforms.DropKey('annotation'),
    transforms.ToTensor('image'),
    transforms.ToTensor('mask', preserve_range=True),
)
