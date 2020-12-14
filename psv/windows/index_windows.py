from collections import namedtuple

import PIL.Image
import shapely.geometry
import tqdm
import xmltodict
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from psv import Annotation
from psv.ptdataset import PsvDataset

import os
import psv.config

class Options:
    DATA_ROOT = psv.config.C.DATA_ROOT



BBox = namedtuple('BBox', 'xmin ymin xmax ymax')

class WindowData:
    def __init__(self, **kwargs):

        self.id = kwargs.pop('id')
        self.annotation_path = kwargs.pop('annotation_path')
        self.image_path = kwargs.pop('image_path')
        self.shape : Polygon = kwargs.pop('shape')
        self.bounds = BBox(*self.shape.bounds)
        self.occluded = kwargs.pop('occluded')
        self.kwargs = dict(kwargs)
        self._image = None

    def width(self):
        return  self.bounds.xmax - self.bounds.xmin

    def height(self):
        return self.bounds.ymax - self.bounds.ymin

    def top(self):
        return self.bounds.ymin

    def left(self):
        return self.bounds.xmin

    def right(self):
        return self.bounds.xmax

    def bottom(self):
        return self.bounds.ymax

    @property
    def image(self):
        if self._image is None:
            self._image =  PIL.Image.open(self.image_path)
        return self._image

    def cropped_image(self, padding=None, pad_units=1):
        xmin, xmax, ymin, ymax = self.bbox



class WindowSet:
    def __init__(self):
        self.id_to_window = {}
        self.windows = []
        self.ds = PsvDataset()

    def index_windows(self):
        for i in tqdm.trange(len(ds), desc="Crawling through annotations"):
            a: Annotation = ds.get_annotation(i)
            for w in a.windows():
                record = dict(w)
                record['annotation_path'] = os.path.relpath(a.annotation_path, start=Options.DATA_ROOT)
                record['image_path'] = os.path.relpath(a.image_path, start=Options.DATA_ROOT)
                record['shape'] = a.shape(w)
                self.id_to_window[record['id']] = record
                self.windows.append(record)

    def add_xnview_keywords(self, xnview_xml):
        """Parse image tags generted with XnView"""
        data = xmltodict.parse(open(xnview_xml, 'rb'), force_list=['Keywords'])
        list = data['XnView_Catalog']['FileList']['File']


if __name__ == '__main__':
    windows = WindowSet()
    windows.index_windows()
    pass
