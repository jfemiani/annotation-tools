"""PyTorch Dataset(s) for the PSV data. 


"""
import torch
import torchvision
import os
from glob import glob
from psv import Annotation
from PIL import Image#, ImageDraw

from aggdraw import Draw, Brush, Pen
#from PIL.ImageDraw2 import Draw, Brush, Pen

import numpy as np
from psv.config import L, C
from torchvision.transforms import functional as F
from torch.utils.data import Dataset


class Compose():
    """Compose functionals that return structured data as dicts. 
    """
    def __init__(self, *ops):
        self.ops = ops
    
    def __call__(self, **kwargs):
        for op in self.ops:
            kwargs = op(**kwargs)
        return kwargs

class ToSegmentation():
    """ Convert our annotation object to a segmentation mask
    """
    def __init__(self, edge_width=5, edge_label='unknown', labels=None):
        self.edge_width = edge_width
        self.edge_label = edge_label
        self.labels = labels

    def __call__(self, **kwargs):
        kwargs['mask'] = _make_segmentation_mask(kwargs['annotation'], 
                                                 self.edge_width,
                                                 self.edge_label, 
                                                 self.labels)
        return kwargs

class ToTensor():
    """ Convert a parameter to a tensor. 
    
    Parameters:
    - *key* -- which named parameter to convert to a tensor. 
    - *preserve_range* -- whether to preserve the range of values, e.g. for segmentation masks
    """
    def __init__(self, key='image', preserve_range=False):
        self.key = key
        self.preserve_range = preserve_range

    def __call__(self, **kwargs):
        a = kwargs[self.key]
        if self.preserve_range:
            a = torch.from_numpy(np.array(a, copy=False))
        else:
            a = F.to_tensor(a)
        kwargs[self.key] = a
        return kwargs   

class DropKey():
    """Remove an field from the data returned from the dataset
    """
    def __init__(self, key):
        self.key = key

    def __call__(self, **kwargs):
        kwargs.pop(self.key)
        return kwargs

class KeyTransform():
    """ Apply a traditional transform to a specific field

    This allows me to mix and match functions that operate on 
    individual fields. 

    Example:
    ```
        KeyTransform('image', ToTensor())
    ```
    """
    def __init__(self, key, func):
        self.key = key
        self.func = func

    def __call__(self,**kwargs):
        arg = kwargs[self.key]
        newarg = self.func(arg)
        kwargs[self.key] = newarg
        return kwargs

class SetCropToFacades():
    def __init__(self, pad=20,
                       pad_units='percent', 
                       skip_unlabeled=True,
                       minsize=None,
                       roikey='crop'):
        self.pad = pad
        self.pad_units = pad_units
        self.skip_unlabeled = skip_unlabeled
        self.minsize=minsize
        self.roikey = roikey
        self.max_IoU_unlabeled = 0.5
    
    def __call__(self, **kwargs):
        annotation = kwargs['annotation']
        width = annotation.image.width
        heigh = annotation.image.height

        valid = []
        if self.skip_unlabeled:
            facades = annotation.iter_objects('facade')
            unlabeled = [annotation.shape(u) for u in annotation.iter_objects('unlabeled')]

           
            for f in facades:
                fs = annotation.shape(f)
                keepit = True
                for u in unlabeled: 
                    iou_unlabeled = fs.intersection(u).area / fs.union(u).area
                    if iou_unlabeled >= self.max_IoU_unlabeled :
                        keepit = False
                        break
                if keepit:
                    valid.append(f)
        else:
            valid = list(annotation.iter_objects('facade'))

        if len(valid) > 0:
            j, i, j2, i2 = annotation.bbox(valid, pad=self.pad, pad_units=self.pad_units)
            w, h = i2-i, j2-j
        else:
            j, i, h, w = 0, 0, annotation.image.width, annotation.image.height
        roi = i, j, h, w

        if self.minsize is not None:
            # Expand the ROI around its center to fit minsize
            # This may go beyond the 0--width, 0--height range of the image

            i, j, w, h = roi  # Indices of the bbox fields
            cx = i + w/2
            cy = j + h/2
            w = max(w, self.minsize[0])
            h = max(h, self.minsize[1])
            i = cx - w/2
            j = cy - h/2 
            roi = i, j, h, w
        
        
        kwargs[self.roikey] = roi 
        return kwargs

                
class ApplyCrop():
    """Crop to a previsioly determined bbox.
    The bbox should be dtermined and set as a field in the kwargs before this
    is called. 

    Parameters
    - key:  The field to crop (e.f. 'image', or 'mask')
    - roikey: The field thaty holds the bbox as (xmin, ymin, xmax, ymax). 

    The idea is that you would first use routines that determine a bbox and
    keep it as a field in the kwargs dict, and then you would
    call this to actually perform the crop on the fields that need it. 
    """
    def __init__(self, key, roikey='crop',
                 pad_if_needed=True, fill=0, padding_mode='constant'):
        self.key =  key
        self.roikey = roikey
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode= padding_mode

    
    def __call__(self, **kwargs):
        img = kwargs[self.key]
        i, j, h, w = [int(value) for value in kwargs[self.roikey]]

        # pad the width if needed
        if self.pad_if_needed and j < 0 or img.width < j + w:
            padding = max(-j, j+w - img.width)
            img = F.pad(img, (padding, 0), self.fill, self.padding_mode)
            j += padding
        # pad the height if needed
        if self.pad_if_needed and i < 0 or img.height < i + h:
            padding= max(-i, i+h - img.height)
            img = F.pad(img, (0, padding), self.fill, self.padding_mode)
            i += padding

        kwargs[self.key] = F.crop(img, i, j, h, w)
        return kwargs

class SetRandomCrop():
    """ Set the 'crop' field to a rectangle contained within the 'image'. 

    This uses the current 'image' to determine the valid set of crop 
    rectangles. It sets the 'crop' field a random bbox (i,j, w, h).

    Once the crop has been set, you can use ApplyCrop('image') to apply
    it to whichever fields you want to crop. 

    Parameters
    - width -- the width of the cropped image
    - height -- the height of the cropped image
    - roikey -- the field name for the crop parameters
    - imagekey -- the field name for the image
    """
    def __init__(self, width=512, height=512, 
                 roikey='crop', imagekey='image'):
        self.width = width
        self.height = height
        self.roikey = roikey
        self.imagekey = imagekey

    def __call__(self, **kwargs):
        image = kwargs[self.imagekey]

        w = self.width
        h = self.height
        if h < image.height:
            i = np.random.randint(0, image.height-h)
        else:
            i = 0
            
        if w < image.width:
            j = np.random.randint(0, image.width-w)
        else:
            j = 0
        kwargs[self.roikey] = (i, j, h, w)
        return kwargs

class Resize():
    """Resize a PIL image to acheive a target width or height
    """
    def __init__(self, key, height=None, width=None, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.key = key
        self.interpolation = interpolation
        assert self.height is not None or self.width is not None

    
    def __call__(self, **kwargs):
        w, h = self.width, self.height
        im = kwargs[self.key]

        if h is None: 
            size = (w, int(round(im.height*w / im.width)))
        elif w is None:
            size = (int(round(im.width * h / im.height)), h)
        else:
            size = (w/im.width, h / im.height)
        
        kwargs[self.key] = im.resize(size, self.interpolation)
        return kwargs

        
class PsvDataset(Dataset):
    def __init__(self,
                 split=None,
                 root=C.DATA_ROOT,
                 transform=None,
                 download=True,
                 url=C.DATA_URL,
                 pwd:bytes=None
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
            xmls = glob(f'{self.root}/Annotations/**/*.xml', recursive=True)
        else:
            xmls = [line.strip() for line in open(f'{self.root}/{split}.txt')]
        self.xmls = xmls 

        if transform is None:
            transform = ToSegmentation()
        self.transform = transform

    def get_annotation(self, index):
        return Annotation(self.xmls[index])

    def __len__(self):
        return len(self.xmls)

    def __getitem__(self, index:int):
        """
        :param index:
        :return:  A tuple with the input and targets
        """
        a = self.get_annotation(index)
        im = a.image
        
        result = self.transform(image=im, annotation=a)

        return result




def _fill_annotations(a: Annotation, label: str, color=0xFF, out=None):
    if out is None:
        out = Image.new(mode='L', size=(Annotation.image.size))
    
    brush = Brush(color)

    d = Draw(out)
    d.setantialias(False)

    for o in a.iter_objects(label):
        xy = a.points(o)
        d.polygon(xy.flatten(), brush)
    
    return d.flush()

def _stroke_annotations(a: Annotation, label: str, color=0xFF, width=5, out=None):
    if out is None:
        out = Image.new(mode='L', size=(Annotation.image.size))
    
    pen = Pen(color, width)

    d = Draw(out)
    d.setantialias(False) # Should not antialias masks

    for o in a.iter_objects(label):
        xy = a.points(o)
        d.polygon(xy.flatten(), pen)
    
    return d.flush()

def _make_segmentation_mask(a: Annotation, edge_width=0, edge_label='unlabeled', labels=None, out=None):
    """Generate a semantic segmentation mask based on labels in z-order.

    The labels should be sorted, with label 0 at the bottom.
    
    Labels are drawn in order, and objects of each label are drawn in the order they occur inthe file.  
    
    If edge_width is > 0 then edges are rendered with 'edge_label'
    """
    names = a.names.tolist()

    if labels is None:
        labels = names
    
    try: 
        edge_color = names.index(edge_label)
    except ValueError:
        # May have passed the label as a number
        edge_color = int(edge_color)
    
    if out is None:
        unknown = names.index('unknown') # 'negative'
        out = Image.new(mode='L', size=(a.image.size), color=unknown)
    
    for label in labels:
        fill_color=names.index(label)
        _fill_annotations(a, label, fill_color, out)
        if edge_width > 0:
            _stroke_annotations(a, label, edge_color, edge_width, out)

    return out

def _make_mask(ds, i, outdir='extracts'):
    import numpy as np 
    a = ds.get_annotation(i) 
    bn = os.path.basename(a.image.filename) 
 
    mask0 = _make_segmentation_mask(a) 
    os.makedirs(f'{outdir}/mask0', exist_ok=True)
    np.savez(f'{outdir}/mask0/{bn.replace("jpg", "npz")}', mask0) 
    Image.fromarray(a.colors[mask0]).save(f'{outdir}/mask0/{bn}') 
 
    mask5 =_make_segmentation_mask(a, edge_width=5) 
    os.makedirs(f'{outdir}/mask5', exist_ok=True)    
    np.savez(f'{outdir}/mask5/{bn.replace("jpg", "npz")}', mask5) 
    Image.fromarray(a.colors[mask5]).save(f'{outdir}/mask5/{bn}') 


def download_images(url=None, root=None, zipname='gsv-images.zip', pwd:bytes=None):
    if url is None:
        url = C.DATA_URL
    if root is None:
        root = C.DATA_ROOT

    import wget
    wget.download(url, zip_path)
    
    extract_images(root, zipname, pwd)
    

def extract_images(root=None, zipname='gsv-images.zip', pwd:bytes=None):
    import zipfile
    import tqdm as tq

    if root is None:
        root = C.DATA_ROOT

    zip_path =  os.path.join(C.DATA_ROOT, 'gsv-images.zip') 
    with zipfile.ZipFile(zip_path) as zf:
        # Loop over each file
        for file in tq.tqdm(zf.namelist(), total=len(zf.namelist())):
            zf.extract(member=file, path=root, pwd=pwd)


TFM_SEGMENTATION = Compose(
    ToSegmentation(), 
    DropKey('annotation'),
    ToTensor('image'),
    ToTensor('mask', preserve_range=True),
    )  

TFM_SEGMENTATION_CROPPED = Compose(
    ToSegmentation(), 

    # Crop in on the facades
    SetCropToFacades(pad=20, pad_units='percent', skip_unlabeled=True, minsize=(512, 512)),
    ApplyCrop('image'), 
    ApplyCrop('mask'),
    
    # Resize the height to fit in the net (with some wiggle room)
    Resize('image', height=600),
    Resize('mask', height=600, interpolation=Image.NEAREST),

    # Reandomly choose a subimage
    SetRandomCrop(512, 512),
    ApplyCrop('image'), 
    ApplyCrop('mask'),
    
    DropKey('annotation'),
    ToTensor('image'),
    ToTensor('mask', preserve_range=True),
    )     