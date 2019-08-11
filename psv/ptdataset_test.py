import os
import nose
from psv.config import L, C
import PIL
import numpy as np
import torch

def test_psv_dataset_segmentation():
    from psv.ptdataset import PsvDataset, ToSegmentation

    ds = PsvDataset(transform=ToSegmentation())

    assert len(ds) == 956, "The dataset should have this many entries"

    mb = ds[0]
    image = mb['image']
    mask = mb['mask']
    assert os.path.basename(image.filename) == 'chicago_illinois-000089-000011-t7jl9DdBqV7NUMA-SmNg0Q.jpg'
    assert isinstance(mask, PIL.Image.Image)
    assert mask.size == image.size
    assert np.sum(mask) == 16233988
    assert np.sum(image) == 1230147282


def test_psv_dataset_tensors():
    from psv.ptdataset import PsvDataset, ToSegmentation, ToTensor, Compose, DropKey

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
    assert torch.sum(mask) == 16233988

    # When converting images to tensor, torch also 
    # adjusts the range to be from 0-1. 
    # This changes the sum.
    assert int(torch.sum(image)) == 4824271


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
    assert torch.sum(mask) == 16233988

    # When converting images to tensor, torch also 
    # adjusts the range to be from 0-1. 
    # This changes the sum.
    assert int(torch.sum(image)) == 4824271


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


    TFM_SEGMENTATION_CROPPED = P.Compose(
        P.ToSegmentation(), 

        # Crop in on the facades
        P.SetCropToFacades(pad=20, pad_units='percent', skip_unlabeled=True, minsize=(512, 512)),
        P.ApplyCrop('image'), 
        P.ApplyCrop('mask'),
        
        # Resize the height to fit in the net (with some wiggle room)
        # THIS is the test case -- the crops will not usually fit anymore
        P.Resize('image', height=400),
        P.Resize('mask', height=400, interpolation=P.Image.NEAREST),

        # Reandomly choose a subimage
        P.SetRandomCrop(512, 512),
        P.ApplyCrop('image'), 
        P.ApplyCrop('mask', fill=24), # 24 should be unlabeled
        
        P.DropKey('annotation'),
        P.ToTensor('image'),
        P.ToTensor('mask', preserve_range=True),
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
    from psv.ptdataset import ToSegmentation, Compose
    c= Compose(ToSegmentation())
    assert len(c.ops) == 1
    assert isinstance(c.ops[0], ToSegmentation)

def test_compose_zero_items():
    from psv.ptdataset import ToSegmentation, Compose
    c= Compose()
    assert len(c.ops) == 0
    assert c(a = 1, b=2) == dict(a=1, b=2)
    