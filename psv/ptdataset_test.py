import os
import nose

import psv.transforms
from psv.config import L, C
import PIL
import numpy as np
import torch

def test_psv_dataset_segmentation():
    from psv.ptdataset import PsvDataset
    from psv.transforms import ToSegmentation

    ds = PsvDataset(transform=ToSegmentation())

    assert len(ds) == 956, "The dataset should have this many entries"

    mb = ds[0]
    image = mb['image']
    mask = mb['mask']
    assert os.path.basename(image.filename) == 'acoruna_spain-000004-000045-22WQvYMnjiiDd_DAIXc-9g.jpg'
    assert isinstance(mask, PIL.Image.Image)
    assert mask.size == image.size
    assert np.sum(mask) == 41435686
    assert np.sum(image) == 882643951


def test_psv_dataset_tensors():
    from psv.ptdataset import PsvDataset
    from psv.transforms import DropKey
    from psv.transforms import ToTensor
    from psv.transforms import ToSegmentation
    from psv.transforms import Compose

    ds = PsvDataset(transform=Compose(
        ToSegmentation(), 
        DropKey('annotation'),
        ToTensor('image'),
        ToTensor('mask', preserve_range=True),
        ))

    assert len(ds) == 956, "The dataset should have this many entries"

    mb = ds[0]
    image = mb['image']
    mask = mb['mask']
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)

    assert mask.shape[-2:] == image.shape[-2:]
    assert torch.sum(mask) == 41435686

    # When converting images to tensor, torch also 
    # adjusts the range to be from 0-1. 
    # This changes the sum.
    assert int(torch.sum(image)) == 3461357


def test_psv_dataset_tfm_segmentation():
    from psv.ptdataset import PsvDataset, TFM_SEGMENTATION

    ds = PsvDataset(transform=TFM_SEGMENTATION)

    assert len(ds) == 956, "The dataset should have this many entries"

    mb = ds[0]
    image = mb['image']
    mask = mb['mask']
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)

    assert mask.shape[-2:] == image.shape[-2:]
    assert torch.sum(mask) == 41435686

    # When converting images to tensor, torch also 
    # adjusts the range to be from 0-1. 
    # This changes the sum.
    assert int(torch.sum(image)) == 3461357


def test_psv_dataset_tfm_segmentation_cropped():
    from psv.ptdataset import PsvDataset, TFM_SEGMENTATION_CROPPED

    ds = PsvDataset(transform=TFM_SEGMENTATION_CROPPED)

    assert len(ds) == 956, "The dataset should have this many entries"

    mb = ds[0]
    image = mb['image']
    mask = mb['mask']
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)

    assert mask.shape[-2:] == image.shape[-2:]

    # Hard to test due to randomness....

    PLOT=True
    if PLOT:
        from matplotlib.pylab import plt
        import torchvision.transforms.functional as F
        a = ds.get_annotation(0)
        plt.figure()
        plt.suptitle('Visualizing test_psv_dataset_tfm_segmentation_cropped, close if ok')
        plt.subplot(121)
        plt.imshow(F.to_pil_image(image))
        plt.title('image')

        plt.subplot(122)
        plt.imshow(a.colors[mask.numpy()])
        plt.title('mask')
        plt.show()

    
def test_psv_dataset_crop_and_pad():
    import psv.ptdataset as P


    TFM_SEGMENTATION_CROPPED = psv.transforms.Compose(
        psv.transforms.ToSegmentation(),

        # Crop in on the facades
        psv.transforms.SetCropToFacades(pad=20, pad_units='percent', skip_unlabeled=True, minsize=(512, 512)),
        psv.transforms.ApplyCrop('image'),
        psv.transforms.ApplyCrop('mask'),
        
        # Resize the height to fit in the net (with some wiggle room)
        # THIS is the test case -- the crops will not usually fit anymore
        psv.transforms.Resize('image', height=400),
        psv.transforms.Resize('mask', height=400, interpolation=P.Image.NEAREST),

        # Reandomly choose a subimage
        psv.transforms.SetRandomCrop(512, 512),
        psv.transforms.ApplyCrop('image'),
        psv.transforms.ApplyCrop('mask', fill=24), # 24 should be unlabeled
        
        psv.transforms.DropKey('annotation'),
        psv.transforms.ToTensor('image'),
        psv.transforms.ToTensor('mask', preserve_range=True),
        ) 

    ds = P.PsvDataset(transform=TFM_SEGMENTATION_CROPPED)

    assert len(ds) == 956, "The dataset should have this many entries"

    mb = ds[0]
    image = mb['image']
    mask = mb['mask']
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)

    assert mask.shape[-2:] == image.shape[-2:]

    # Hard to test due to randomness....

    PLOT=True
    if PLOT:
        from matplotlib.pylab import plt
        import torchvision.transforms.functional as F
        a = ds.get_annotation(0)
        plt.figure()
        plt.suptitle('Visualizing test_psv_dataset_tfm_segmentation_cropped,\n'
                     'close if ok \n '
                     'confirm boundary is  marked unlabeled')
        plt.subplot(121)
        plt.imshow(F.to_pil_image(image))
        plt.title('image')

        plt.subplot(122)
        plt.imshow(a.colors[mask.numpy()])
        plt.title('mask')
        plt.show()

         

def test_compose_one_item():
    from psv.transforms import ToSegmentation
    from psv.transforms import Compose
    c= Compose(ToSegmentation())
    assert len(c.ops) == 1
    assert isinstance(c.ops[0], ToSegmentation)

def test_compose_zero_items():
    from psv.transforms import ToSegmentation
    from psv.transforms import Compose
    c= Compose()
    assert len(c.ops) == 0
    assert c(a = 1, b=2) == dict(a=1, b=2)
    