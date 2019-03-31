import json
import shutil

from PIL import Image
import numpy as np
from matplotlib.axes import Axes
from numpy import array, eye
from numpy.linalg import inv
import os
import shapely
import shapely.geometry
from easydict import EasyDict
import xmltodict
from matplotlib.patches import Polygon
from matplotlib.text import Text
from matplotlib import pyplot as plt
from ast import literal_eval

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'psvdata'))


# class AnnotatedPolygon(object):
#     def __init__(self, username, pts):
#         self.username = username
#         self.pts = np.array[[(float(p['x']), float(p['y'])) for p in pts]]
#
#     @property
#     def x(self): return self.pts[:, 0]
#
#     @property
#     def y(self): return self.pts[:, 1]
#
#     @property
#     def xy(self): return self.pts
#
#     def as_dict(self):
#         return dict(
#             username=self.username,
#             pts = [dict(x=x, y=y) for (x, y) in self.pts]
#         )
#
#
# class AnnotatedObject(object):
#     def __init__(self, name, deleted, verified, attributes, type, polygon, parts):
#         self.name = name
#         self.deleted = deleted
#         self.verified = verified
#         self.attributes = attributes
#         self.type = type
#
#         if isinstance(polygon, AnnotatedPolygon):
#             self.polygon = polygon
#         else:
#             self.polygon = AnnotatedPolygon(**polygon)
#
#         self.parts = parts


class Annotation(object):
    def __init__(self, annotation=None, dat=None, root=DATA_ROOT, roots=None):

        # The JSON file for a rectified subimage
        if roots is None:
            roots = []
        self.dat = dat

        # The root folder for the dataset
        self.root = root

        # The names used in the dataset are kept in a text file to make them easier to edit
        # The line number in the text file is the z-order of the label
        self.names = np.loadtxt(self._path('names.txt'), dtype='str')
        self.z_orders = {name: i for (i, name) in enumerate(self.names, 0)}

        # The colors were generated using Glasbey's categorical color choosing method
        # to make 24 maximally distinct colors. 
        self.colors = np.loadtxt(self._path('colors.txt'), delimiter=',', dtype=np.uint8)

        if annotation:
            self.annotation_path = annotation
        elif dat:
            self.annotation_path = dat.annotation


        # It is possible that the annotation file does not exist
        self.has_annotation = os.path.isfile(self.annotation_path)

        # Try harder to find the annotation
        for root in [root] + roots:
            if not self.has_annotation:
                filename = os.path.basename(self.annotation_path)
                folder = os.path.dirname(self.annotation_path)
                folder = os.path.basename(folder)
                path = os.path.join(root, 'Annotations', folder, filename)
                if os.path.isfile(path):
                    self.has_annotation = True
                    self.annotation_path = path
                    break

        # Some JSON files were never exported for some reason -- so there may not be
        # an associated annotation XML
        if self.has_annotation:
            a = EasyDict(xmltodict.parse(open(self.annotation_path, 'rb'),
                                         force_list=('object', 'pt')))
            for o in a.annotation.object:
                # Boolean-ish objects are tricky to handle
                o.deleted = literal_eval(o.deleted)
                o.verified = literal_eval(o.verified)

            self.annotation = a.annotation
        else:
            self.annotation = None

        # Get the image associated with annotation (if we can find it)
        self.image_path = None
        if self.annotation:
            for r in [self.root] + roots:
                image_path = os.path.join(r,
                                          'Images',
                                          self.annotation.folder,
                                          self.annotation.filename)
                if os.path.isfile(image_path):
                    self.image_path = image_path
                    break

        if self.image_path is None:
            self.image_path = self.annotation_path.replace('Annotations', 'Images').replace('.xml', '.jpg')

        # Open the image (header, using PIL) if it exists. 
        if os.path.isfile(self.image_path):
            self.image = Image.open(self.image_path)
        else:
            self.image = None

        # Originally this was just for a rectified facade subimage, but now I am
        # using the class for an entire facade image.

        if dat is None:
            # The data is in a file alongside the image
            dat_file = self.image_path.replace('-highlighted', '')      # Will not have '-highlighted' in the name
            dat_file = dat_file.replace('.jpg', '.json')  # will not be a jpg
            if os.path.isfile(dat_file):
                with open(dat_file, 'rb') as df:
                    dat = EasyDict(json.load(df))

        # 'dat' will be the contents of the JSON file if this was a sub-image
        #  or None if this is an entire images
        if dat:
            self.projection = array(dat.projection)
            self.translation = array([[1, 0, dat.subimage.top],  # When I saved the JSON files,
                                      [0, 1, dat.subimage.left],  # I mixed op top and left
                                      [0, 0, 1]])
            self.extent=[dat.subimage.bottom-0.5,
                         dat.subimage.top-0.5,
                         dat.subimage.left-0.5,
                         dat.subimage.right-0.5]
        else:
            # Not a subimage -- so no projection or translation.
            # Since there is no JSON file, the annotation argument must be passed in
            assert annotation is not None, "You must use either `dat` or `annotation` arguments"
            self.projection = eye(3)
            self.translation = eye(3)
            if self.image:
                self.extent = [-0.5,
                               self.image.width-0.5,
                               self.image.height - 0.5,
                               -0.5]
            else:
                self.extent = [0 ,1, 1, 0]

        # The projection matrices to rectify (or restore) points
        self.rectify_matrix = inv(self.projection) @ inv(self.translation)
        self.rectify_inverse_matrix = self.translation @ self.projection

    def _path(self, *args):
        return os.path.join(self.root, *args)

    def iter_objects(self, label=None):

        # The label argument is now (potentially) a set of labels to show.
        # Set operations can be used to select which labels to retrieve.

        if label is None:
            label = [str(name) for name in self.names]
        elif isinstance(label, str):
            label = [label]

        return (o for o in self.annotation.object if o.name in label)

    def __iter__(self):
        return self.iter_objects()

    def points(self, object, tfm=None):
        # Get the points from the polygon associated with the object
        if isinstance(object, EasyDict):
            return self.points(object.polygon.pt)

        if isinstance(object, list):
            pts = array([(float(p.x), float(p.y)) for p in object])
        elif isinstance(object, np.ndarray):
            pts = object
        else:
            assert False, f"Invalid argument for 'object', ({object})"

        # Do a homogeneous transform on points (expects no points-at-infinity)
        if tfm is not None:
            pts = pts @ tfm.T[:2, :] + tfm.T[2, :]
            pts = pts[:, :2] / pts[:, 2, None]

        return pts

    def shape(self, object):
        return shapely.geometry.Polygon(self.points(object))

    def rectified(self, pts):
        # Returns a rectified copy of pts
        assert pts is not None
        pts = self.points(pts)
        pts = pts @ self.rectify_matrix.T[:2, :] + self.rectify_matrix.T[2, :]
        pts = pts[:, :2] / pts[:, 2, None]
        return pts

    def unrectified(self, pts):
        assert pts is not None
        pts = self.points(pts)
        pts = pts @ self.rectify_inverse_matrix.T[:2, :] + self.rectify_inverse_matrix.T[2, :]
        pts = pts[:, :2] / pts[:, 2, None]
        return pts

    def add_unlabeled_facades(self, width, min_dist=20):
        """If a facade is near the edge,
        add an 'unlabeled' region that overlaps it. 
        This means that the absence of a positive label is unknown,
        rather than negative. We need to do this because the rectification 
        script did not export these facades. 
        
        :param width:
            Width of the image.
        :param min_dist:
            The minimum distance of a facade to the edge before
            we consider it unlabeled.
        """
        facades = list(self.iter_objects('facade'))
        for fac in facades:
            pts = self.rectified(fac.polygon.pt)
            left = min(pts[:, 0])
            right = max(pts[:, 0])
            unlabeled = left < min_dist or (width - right) < min_dist
            if unlabeled:
                unk = EasyDict(fac.copy())
                unk.name = 'unlabeled'
                unk.id = len(self.annotation.object)
                self.annotation.object.append(unk)

    def find_duplicates(self, max_iou=0.7):
        obs = self.annotation.object
        for i in range(len(obs)):
            if obs[i].deleted:
                continue

            for j in range(i + 1, len(obs)):
                if obs[j].deleted:
                    continue

                # To ba a duplicate, they must share the same label
                if obs[i].name != obs[j].name:
                    continue

                poly_i = self.unrectified(obs[i].polygon.pt)
                poly_j = self.unrectified(obs[j].polygon.pt)

                # Some degenerate objects exist in the data somehow.
                # A polygon must have at least 3 points
                if (len(poly_j) < 3) or (len(poly_i) < 3):
                    continue

                # A polygon must have real numbers as coordinates
                if np.isnan(poly_j).any() or np.isnan(poly_i).any():
                    continue

                # Some polygons have overlapping edges.
                # shapely has a `buffer(0)` that removes degenerate parts.
                shape_i = shapely.geometry.Polygon(poly_i).buffer(0)
                shape_j = shapely.geometry.Polygon(poly_j).buffer(0)

                iou = shape_i.intersection(shape_j).area / shape_i.union(shape_j).area
                if iou > max_iou:
                    yield i, j

    def save_annotation(self,
                        path: str=None,
                        filename: str=None,
                        folder: str=None,
                        root: str=DATA_ROOT,
                        copy_image: bool=False,
                        image_path: str=None,
                        mkdirs: bool=False):
        """ Save the annotations (e.g. after merging or editing)
        
        This saves the annotations for an image. 
        The image should be stored at the typical LabelMe location:
        'root/Images/folder/filename.replace('.xml', '.jpg')'

        :param path: The explicit path to the annotation (overrides the default
        based on 'root/Annotations/folder/filename')
        :param filename: The filename of the image or XML
        :param folder: The folder of the image or XML
        :param root: The root folder of the dataset
        :param copy_image: Copy the image to a parallel folder, if it is not already there.
        :param image_path: The location to save the image. Defaults to 'root/Images/<folder>/filename.jpg'
        :param mkdirs: If true, make the directories for `path` or `image_path` if the dont already exist.
        """

        # Use either the argument or the current folder.
        if folder is None:
            folder = self.annotation.folder

        # Use either the arg of current filename. 
        if filename is None:
            filename = self.annotation.filename

        # Make sure we have a folder to save our results to. Otherwise an exception will happen
        if mkdirs:
            os.makedirs(os.path.join(root, 'Annotations', folder), exist_ok=True)
            if copy_image:
                os.makedirs(os.path.join(root, 'Images', folder), exist_ok=True)

        # We do not modify our annotation in place -- we will make a copy
        a = EasyDict(self.annotation.copy())
        a.folder = folder
        a.filename = filename

        # The location of the XML file
        if path is None:
            path = os.path.join(root, 'Annotations', folder, filename.replace('.jpg', '.xml'))

        # If we want to copy the image, we need to calculate the old and new paths
        if copy_image:
            if image_path is None:
                image_path = os.path.join(root,
                                          'Images',
                                          self.annotation.folder,
                                          self.annotation.filename.replace('.xml', '.jpg'))
            new_image_path = os.path.join(root,
                                          'Images',
                                          folder,
                                          filename.replace('.xml', '.jpg'))

            # Do not overwrite if the image is already in the destination
            # (perhaps we should warn once)
            if not os.path.isfile(new_image_path):
                shutil.copy(image_path, new_image_path)

        # Fix boolean-ish values
        for o in self.annotation.object:
            o.deleted = int(o.deleted)
            o.verified = int(o.verified)

        # Save the copied XML to its target location
        with open(path, 'w') as f:
            xmltodict.unparse({'annotation': a}, f)

    def remove_deleted(self):
        """ Remove deleted objects.

        The XML format included a 'deleted' flag for objects.
        The format allows deleted objects to remain in the file.
        This can be very confusing -- I always forget that deleted objects will be present.
        This function purges those deleted objects from the file.

        """
        self.annotation.object = [o for o in self.annotation.object if not literal_eval(o.deleted)]

    def bbox(self, objects=None, pad=0, pad_units='prop'):
        """Get the minmax box (xmin, ymin, xmax, ymax)
        
        :param objects: Iterable or a single object (a dict)
        :param pad: The amount to pad the bbox by (often we want some padding for context)
        :param pad_units: The units to pad with: can be 'prop' or 'pixels'. If 'prop'
                    is is proportional of the max over width and height.
        """

        # Allow this to be used to a single object
        if isinstance(objects, dict):
            objects = [objects]

        xmin = self.image.width
        ymin = self.image.height
        xmax = 0
        ymax = 0
        for o in objects:
            pts = self.unrectified(o.polygon.pt)
            pts_min = pts.min(0)
            pts_max = pts.max(0)
            xmin = min(xmin, pts_min[0])
            ymin = min(ymin, pts_min[1])
            xmax = max(xmax, pts_max[0])
            ymax = max(ymax, pts_max[1])

        # Pad
        try:
            pad_x, pad_y = pad
        except TypeError:
            pad_x, pad_y = pad, pad

        if pad_units == 'prop':
            scale = max((xmax - xmin), (ymax - ymin))
            pad_x *= scale
            pad_y *= scale

        return xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y

    def plot(self, ax: Axes = None, show_image=True, labels=None, objects=None, alpha=0.6, ls='-'):
        # Reuse the current axis if none was passed in
        ax = ax or plt.gca()

        # I am now using this same class for the ENTIRE image as well
        # as for sub images of facades.
        if show_image:
            if self.extent:
                ax.imshow(self.image, zorder=-1, extent=self.extent)
            else:
                ax.imshow(self.image, zorder=-1)

        if self.extent:
            ax.set_xbound(self.extent[0], self.extent[1])
            ax.set_ybound(self.extent[2], self.extent[3])

        # If this is a subimage of a facade, use the JSON file to 
        # determine the actual facade polygon
        if self.dat is not None and self.annotation is None:
            pts = array(self.dat.polygon)
            p = Polygon(pts, color=[0, 0, 0], zorder=self.z_orders['facade'], fill=False, alpha=alpha, hatch='/')
            ax.add_patch(p)
            return

        if objects is None:
            objects = self.iter_objects(labels)
        else:
            objects = objects
        for o in objects:
            n = o.name
            z = self.z_orders[n]
            c = self.colors[z] / 255.

            pts = self.unrectified(o.polygon.pt)

            # Just in case -- I think I have pruned out deleted records but if
            # any remain, render them with a black crosshatch overlaid.  
            if o.deleted:
                ax.add_patch(Polygon(pts, color='k', hatch='+', fill=False, zorder=50))
            else:
                ax.add_patch(Polygon(pts, color=c, edgecolor=None, zorder=z, linestyle=ls, alpha=alpha))
                ax.add_patch(Polygon(pts, color=c, lw=2, linestyle=ls, fill=False, zorder=50))
            ax.add_artist(Text(*pts.mean(0), o.name, zorder=55))


def test():
    f = Annotation('./psvdata/Annotations/merged/ny_many-0440.xml')
    fig = plt.figure()
    ax = plt.gca()
    plt.imshow(f.image)
    f.plot(ax)

    plt.show()

if __name__ == '__main__':
    test()