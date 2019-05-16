import sys
import os
from easydict import EasyDict
import datetime
from psv.annotation import Annotation
import numpy as np
import tqdm as tq

ROOT = './psvdata'
VERSION = '0.0.0'

only_facades = lambda x: x.name == 'facade'
only_windows = lambda x: x.name == 'window'
only_windows_and_facades = lambda x: x.name == 'window' or x.name == 'facade'
keep_everything = lambda x: True


def convert_to_coco(annotations,
                    root=None,
                    filt=None,
                    cocoroot=None,
                    tqdmargs={}):
    if root is None:
        root = ROOT
    assert os.path.isdir(root)

    if cocoroot is None:
        cocoroot = f'{root}/coco'
    os.makedirs(cocoroot, exist_ok=True)

    if filt is None:
        filt = keep_everything

    now = datetime.datetime.now()

    cocodata = EasyDict()

    # Info
    cocodata.info = EasyDict()
    cocodata.info.year = now.year
    cocodata.info.version = VERSION
    cocodata.info.description = 'Annotations for GSV images'
    cocodata.info.contributer = 'KAUST, and Miami University (Ohio)'
    cocodata.info.url = 'https://github.com/jfemiani/annotation-tools'
    cocodata.info.datecreated = now.isoformat()

    # Categories
    names = np.loadtxt(f"{root}/names.txt", dtype=str)
    name2id = {}
    cocodata.categories = []
    for (i, name) in enumerate(names, 1):
        cat = EasyDict()
        cat.supercategory = 'outdoor'
        cat.id = i
        cat.name = str(name)
        name2id[name] = i
        cocodata.categories.append(cat)

    # Licenses
    cocodata.license = EasyDict()
    cocodata.license.id = 0
    cocodata.license.name = "TBD -- not public yet"
    cocodata.license.url = "TBD -- do not use this until resolved"

    # Images & Annotations
    cocodata.images = []
    cocodata.annotations = []

    for annotation_path in tq.tqdm(annotations, **tqdmargs):
        a = Annotation(annotation_path)
        objects = [o for o in a.iter_objects() if filt(o)]

        if len(objects) == 0: continue

        image = EasyDict()
        image.id = len(cocodata.images)
        image.width = a.image.width
        image.height = a.image.height
        image.file_name = os.path.relpath(a.image.filename, start=root)
        image.license = 0
        image.flickr_url = None
        image.coco_url = None
        image.date_captures = None
        cocodata.images.append(image)

        for o in objects:
            if o.deleted:
                continue
            try:
                anno = EasyDict()
                anno.id = len(cocodata.annotations)
                anno.image_id = image.id
                anno.category_id = name2id[o.name]
                x, y, x2, y2 = a.bbox(o)
                anno.bbox = [x, y, x2 - x, y2 - y]
                anno.area = a.shape(o).area
                anno.iscrowd = 0
                anno.segmentation = [a.points(o).flatten().tolist()]
                cocodata.annotations.append(anno)
            except:
                print(f"Bad object: {o.id} in {annotation_path}")

    return cocodata


ALL_NAMES = [name.strip() for name in open('psvdata/names.txt').readlines()]

if __name__ == '__main__':
    from glob import glob
    from random import shuffle, seed
    import json

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        '--filt',
        '-f',
        nargs='*',
        type=str,
        help='Filters to include; all for everything',
        default='all')
    p.add_argument('--slug', type=str, default='filt')
    p.add_argument('--root', type=str, default=os.path.abspath(ROOT))
    p.add_argument('--seed', type=int, default=127)
    p.add_argument('--outdir', type=str, default='coco')

    args = p.parse_args()

    ROOT = args.root

    if args.filt == 'all':
        args.filt = ALL_NAMES
    if args.slug == 'filt':
        args.slug = '-'.join(args.filt)

    os.makedirs(args.outdir, exist_ok=True)

    SLUG = args.slug
    filt = lambda o: o.name in args.filt

    annots = glob(os.path.join(ROOT, 'Annotations/**/*.xml'), recursive=True)

    seed(args.seed)
    shuffle(annots)

    num = len(annots)
    num_train = int(0.8 * num)
    num_val = num - num_train
    train_annots = annots[:num_train]
    val_annots = annots[num_train:]

    train_ds = convert_to_coco(train_annots, filt=filt)
    val_ds = convert_to_coco(val_annots, filt=filt)

    json.dump(train_ds,
              open(os.path.join(args.outdir, f'train-{SLUG}.json'), 'w'))
    json.dump(val_ds, open(os.path.join(args.outdir, f'val-{SLUG}.json'), 'w'))
