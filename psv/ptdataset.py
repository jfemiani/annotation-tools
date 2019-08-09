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

from psv.config import L, C
from torch.utils.data import Dataset

class PsvDataset(Dataset):
    def __init__(self,
                 split=None,
                 root=C.DATA_ROOT,
                ):
        """
        :param split: A split of the data (see the txt files under DATA_ROOT).
        :param root: The root folder of the dataset.
        """

        self.root = root

        if split is None:
            xmls = glob(f'{self.root}/Annotations/**/*.xml', recursive=True)
        else:
            xmls = [line.strip() for line in open(f'{self.root}/{split}.txt')]
        self.xmls = xmls 

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
        
        return im



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

def make_mask(ds, i, outdir='extracts'):
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