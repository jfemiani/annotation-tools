""" Merge rectified annotations into a single (unrectified) image.

During our labeling process, labelers first draw quadrilaterals around facades.
Then each facade is cropped and rectified using a homography.
The sub-facade featureas are marked on the rectified images.


This script reads in a batch of images an un-rectifies them.


To merge in a new set of annotations use this command:

```
    python ingest_annotations.py -u path/to/new/annotations   -f 'premerge'
```

Then inspect the new files to make sure everything is ok

```
    mv gsv24/premerge/Annotations/*.* gsv24/merged/Annotations/*.*
```

"""
import os
import shutil

from tqdm import tqdm, trange
from annotation import Annotation
from glob import glob
from psv import config

# These are folders with messier sets of images... needs organization
DEFAULT_ROOTS = [
    os.path.expanduser('~/Projects/old-work/data/labeling/'),
    os.path.expanduser('~/Projects/old-work/data/labeling/output-2018-12-02/'),
    os.path.expanduser('~/Projects/facade-data/gsv24'),
]

if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('--updates', '-u', type=str,
                   default='updates/Facades_Jan23',
                   help='A folder of new annotation XMLs. Should have a sub-folder named "Annotations"')

    p.add_argument('--root', '-r', type=str, default=config.DEFAULT_ROOT,
                   help='The root folder of the dataset')

    p.add_argument('--path', '-p', type=str, nargs='*',
                   default=DEFAULT_ROOTS,
                   help='The search path for images (other data roots)')

    p.add_argument('--folder', '-f', type=str, #default='merged',
                   help='The sub-folder that will contain the merged annotations ')
    args = p.parse_args()

    if not os.path.isdir(args.updates):
        print("Updates should eb a folder -- perhaps you need to unzip a file?")

    files = glob(os.path.join(args.updates, '*.xml'))

    images = []
    missing = []
    for fn in tqdm(files):
        f = Annotation(fn, root=args.root, roots=args.path)
        images.append(f.image_path)

        # Find the original image  and XML
        #    The per-facade images share the same basename as the merged image, but have a suffix
        #    that starts with the string "-facade-"
        ori_image_basename = os.path.basename(f.image_path).split('-facade-')[0] + '.jpg'
        ori_xml_basename = os.path.basename(f.image_path).split('-facade-')[0] + '.xml'

        # As an artifact of the way things were processed, images may be in several potential locations
        # search through all paths for an image
        ori_image_path = None
        for rt in args.path:
            fns = glob(os.path.join(rt, '**', ori_image_basename), recursive=True)
            if len(fns) > 0:
                ori_image_path = fns[0]
                break

        # Same for the original XML file (the original XML has the facade labels)
        ori_xml_path = None
        for rt in args.path:
            fns = glob(os.path.join(rt, '**', ori_xml_basename), recursive=True)
            if len(fns) > 0:
                ori_xml_path = fns[0]
                break

        # Copy the original XML to the output folder
        merged_xml_path = os.path.join(args.root, args.folder, ori_xml_basename)
        if not os.path.isfile(merged_xml_path):
            shutil.copy_file(ori_xml_path, merged_xml_path)

        # Open the original XML file
        ori_f = Annotation(merged_xml_path, root=args.root, roots=args.path)



        # Much of this code is missing? What happened? Did I not commit from home?