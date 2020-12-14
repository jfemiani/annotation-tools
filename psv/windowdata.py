"""Dataset for jsut the windows"""

import pandas as pd
from zipfile import ZipFile
import ast
from urllib.request import urlretrieve
from PIL import Image
import shapely.geometry
import os
import numpy as np
import matplotlib.pylab as plt
import psv.config


class Options:
    URL = 'http://teams.cec.miamioh.edu/Vision/facades/windows.zip'
    DATA_ROOT = psv.config.DATA_ROOT
    LOCAL_ZIP_PATH = os.path.join(DATA_ROOT, 'windows.zip')
    WINDOWS_CSV = os.path.join(DATA_ROOT, 'windows.csv')


class WindowData:
    def __init__(self, download=True, filename=Options.LOCAL_ZIP_PATH, shuffle=True):
        self.filename = filename
        self.shuffled = shuffle
        if download:
            self.download(filename=filename)
        self.data = ZipFile(self.filename)
        self.dataframe = pd.read_csv(self.data.open('windows.csv', 'r'), index_col='id')
        if shuffle:
            self.dataframe = self.dataframe.sample(frac=1)

    @staticmethod
    def download_zipfile(url=Options.URL, filename=Options.LOCAL_ZIP_PATH, force=False):
        if force or not os.path.isfile(filename):
            _, status = urlretrieve(url, filename=filename)

    def meta(self, index):
        return self.dataframe.iloc[index]

    def image_path(self, index):
        return f'windows/{self.meta(index).name}.jpg'

    def image(self, index):
        return Image.open(self.data.open(self.image_path(index)))

    def polygon(self, index):
        return np.array(ast.literal_eval(self.meta(index).local_polygon))

    def shape(self, index):
        """Shapely shape for the window 
        supports many geometric operations like 'buffer', 'intersect', 'interpolate', etc. """
        return shapely.geometry.Polygon(self.polygon(index))

    def occlusion(self, index):
        return shapely.wkt.loads(self.meta(index).occluding_polygon)

    def plot(self, index, ax=None, **kwargs):
        poly_options = dict(color='red', lw=2, ls='--', fill=False)
        poly_options.update(**kwargs)

        ax = ax or plt.gca()
        image = self.image(index)
        poly = self.polygon(index)
        ax.imshow(image)
        ax.add_patch(plt.Polygon(poly, **poly_options))

    def __getitem__(self, index):
        return self.image(index), self.shape(index)

    def __len__(self):
        return len(self.dataframe)

    def __repr__(self):
        return f'(WindowData with {len(self)} windows, shuffled={self.shuffled})'



