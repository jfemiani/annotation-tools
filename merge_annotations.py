# This script is a tool to maerge in new annotations created by a lebeler at
# a different site.
import os
import shutil

from tqdm import tqdm, trange
from annotation import Annotation
from glob import glob

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
                   help='A folder of new annotation XMLs')

    p.add_argument('--root', '-r', type=str, default='./gsv24',
                   help='The root folder of the dataset')

    p.add_argument('--path', '-p', type=str, nargs='*',
                   default=DEFAULT_ROOTS,
                   help='The search path for images (other data roots)')

    p.add_argument('--folder', '-f', type=str, default='merged',
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

        # Find the original image
        ori_image_basename = os.path.basename(f.image_path).split('-facade-')[0] + '.jpg'
        ori_xml_basename = os.path.basename(f.image_path).split('-facade-')[0] + '.xml'

        ori_image_path = None
        for rt in args.path:
            fns = glob(os.path.join(rt, '**', ori_image_basename), recursive=True)
            if len(fns) > 0:
                ori_image_path = fns[0]
                break

        ori_xml_path = None
        for rt in args.path:
            fns = glob(os.path.join(rt, '**', ori_xml_basename), recursive=True)
            if len(fns) > 0:
                ori_xml_path = fns[0]
                break

        merged_xml_path = os.path.join(args.root, args.folder, ori_xml_basename)
        if not os.path.isfile(merged_xml_path):
            shutil.copy_file(ori_xml_path, merged_xml_path)

        ori_f = Annotation(merged_xml_path, root=args.root, roots=args.path)
