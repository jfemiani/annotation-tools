import numpy as np
import os
import matplotlib.backend_bases
from matplotlib.backend_bases import KeyEvent, ResizeEvent
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Polygon
from glob import glob
from copy import copy, deepcopy
from datetime import datetime
from uuid import uuid1

from matplotlib.transforms import Bbox

import facadesubimage as fsi
from easydict import EasyDict
from labelboxes import LabelBoxes
from polygonselector import PolygonSelector
from polygoneditor import PolygonEditor

DATA_ROOT = './gsv24/'
DEFAULT_LABEL = 24  # unknown


class AnnotationEditor(object):
    """Edit Image Annotations

    """
    canvas: matplotlib.pyplot.FigureCanvasBase
    ax: matplotlib.pyplot.Axes
    fill_alpha: float
    username: str
    last_filter: dict
    poly_selector: PolygonSelector
    poly_editor: PolygonEditor

    def __init__(self, ax, facade=None,
                 root=DATA_ROOT,
                 on_select=None,
                 on_hover=None,
                 fill_alpha=0.4,
                 expand_by=2,
                 username='anonymous',
                 annotations=None,
                 folder='merged'):
        """
        Parameters:
        - *ax*: The axis to render onto, 
        - *facade*: A facade object to edit. 
        - *labels_selector*: An (optional) LabelSelector widget to 
          allow the labels of annotations to be changed. 
        """
        super().__init__()

        if folder is None and facade is not None:
            folder = facade.annotation.folder

        if annotations is None:
            annotations = glob(os.path.join(root, 'Annotations', folder, '*.xml'))

        if facade is None:
            facade = fsi.FacadeSubImage(annotation=annotations[0])

        self.ax = ax
        self.canvas = ax.figure.canvas
        self._facade = None
        self.labels_selector = None
        self.cids = []
        self.fill_alpha = fill_alpha
        self.username = username

        # Keep a list of all of the annotated images so we can load the 'next' file 
        # automatically
        self._annotation_list = copy(annotations)
        self._annotation_index = -1
        if facade.annotation_path in annotations:
            self._annotation_index = annotations.index(facade.annotation_path)

        # The parameters used to filter objects
        self.last_filter = None

        self._label_box = None

        self.axis_image = None

        # Callbacks
        self.on_hover = on_hover
        self.on_select = on_select

        ##
        # Not sure that ths widget should be aware of the root -- should probably be passed
        # the names and colors as parameters....
        self.root = root
        self.names = np.loadtxt(f"{self.root}/names.txt", dtype='str')
        self.colors = np.loadtxt(f"{self.root}/colors.txt", delimiter=',', dtype=np.uint8) / 255.
        self.z_orders = {i: name for (name, i) in enumerate(self.names)}
        ##

        # I have a (hopefully) generic polygon selector that allows me to interact with the
        # polygons
        self.poly_selector = PolygonSelector(self.ax,
                                             onactivate=self._on_poly_activate,
                                             onhover=self._on_poly_hover,
                                             expandby=expand_by)

        # Create a text object to render information about the active object
        # To make the text visible over any background, give it a shadow
        self.label = self.ax.text(0, 0, 'unknown', visible=False,
                                  horizontalalignment='center',
                                  verticalalignment='center')
        self.label.set_path_effects([patheffects.Normal(),
                                     patheffects.SimplePatchShadow(offset=(1, -1), shadow_rgbFace=(1, 1, 0))])

        # Create polygon editor when the user pressed a certain key (e.g. enter)
        self.poly_editor = None
        self._poly_editor_index = -1  # Save the index of the object we last started editing

        # Set our facade using the property setter to trigger updates
        self.facade = facade

        # Keep track of changes made with this tool
        self.history = []
        self.future = []

        # Initialize properties to pass stupid code inspection
        if False and "This is dumb":
            self.active_index = self.get_active_index()
            self.active_object = self.get_active_object()
            self.hover_index = self.get_hover_index()
            self.hover_object = self.get_hover_object()

        self.connect_events()

    def get_facade(self):
        """The facade object we are annotating (not the path to it)"""
        return self._facade

    def set_facade(self, value):
        assert isinstance(value, fsi.FacadeSubImage)

        if value != self._facade:
            self._facade = value
            self.ax.dataLim.set(self.ax.dataLim.null())
            if self.axis_image is not None:
                self.axis_image.remove()
            self.axis_image = self.ax.imshow(value.image, zorder=-1)
            self.ax.figure.suptitle(os.path.relpath(value.annotation_path, start=self.root))

            # Update the object selector
            self.poly_selector.clear()
            for i, o in enumerate(value.annotation.object):
                self.poly_selector.add_polygon(self._make_polygon(o))
                self._update_polygon(i)

    facade = property(get_facade, set_facade)

    def get_active_index(self):
        """Set the active object index"""
        # Delegate selection to self.poly_selector
        return self.poly_selector.active_index

    def set_active_index(self, value):
        self.poly_selector.active_index = value

    active_index = property(get_active_index, set_active_index)

    def get_active_object(self)->EasyDict:
        if self.active_index >= 0:
            result = self.facade.annotation.object[self.active_index]
        else:
            result = None
        return result

    def set_active_object(self, ob):
        if ob is None:
            index = -1
        else:
            index = self.facade.annotations.index(ob)
        self.set_active_index(index)

    active_object = property(get_active_object, set_active_object)

    def get_hover_index(self):
        """Access the 'hover' object that would be selected LMB"""
        return self.poly_selector.hover_index

    def set_hover_index(self, value):
        self.poly_selector.hover_index = value

    hover_index = property(get_hover_index, set_hover_index)

    def get_hover_object(self):
        if self.hover_index < 0:
            return None
        else:
            return self.facade.annotation.object[self.hover_index]

    def set_hover_object(self, ob):
        if ob is None:
            index = -1
        else:
            index = self.facade.annotations.index(ob)
        self.set_hover_index(index)

    hover_object = property(get_hover_object, set_hover_object)

    def get_active_label(self):
        return self.label.get_text()

    def set_active_label(self, value):
        if value != self.active_label:
            self.label.set_text(value)
            self.update()

    active_label = property(get_active_label, set_active_label)

    # For managing history -- we keep a log of actions
    ACTION_LOAD_ANNOTATION = 'load-annotation'
    ACTION_SET_OBJECT = 'set-object'
    ACTION_INSERT_OBJECT = 'insert-object'

    def get_object(self, index):
        """Gets a copy of the object at an index"""
        return deepcopy(self.facade.annotation.object[index])

    def set_object(self, index, value, ori_value=None):
        assert index >= 0

        if ori_value is None:
            ori_value = self.get_object(index)

        self._do(self.ACTION_SET_OBJECT,
                 data=dict(index=index, value=value, ori_value=ori_value))

    def insert_object(self, index, value):
        self._do(self.ACTION_INSERT_OBJECT,
                 data=dict(index=index, value=value))

    def _do(self, action, data, clear_future=True):
        self.history.append(dict(action=action, data=deepcopy(data)))
        if clear_future:
            del self.future[:]

        if action == self.ACTION_LOAD_ANNOTATION:
            filename = data['filename']
            assert 'ori_filename' in data
            self.facade = fsi.FacadeSubImage(filename, root=self.root)
        elif action == self.ACTION_SET_OBJECT:
            index = data['index']
            obj = data['value']
            assert 'ori_value' in data
            self.facade.annotation.object[index] = deepcopy(obj)
            self.active_index = index
            self._update_polygon(index)
        elif action == self.ACTION_INSERT_OBJECT:
            index = data['index']
            obj = data['value']
            self.facade.annotation.object.insert(index, obj)
            self.poly_selector.insert_polygon(index, self._make_polygon(obj))
            self.active_index = index
            self._update_polygon(index)

    def load_annotation(self, filename, ori_filename=None):
        if ori_filename is None:
            ori_filename = self.backup()  # elf.facade.annotation_path
        action = self.ACTION_LOAD_ANNOTATION
        data = dict(filename=filename, ori_filename=ori_filename)
        self._do(action, data)

    def undo(self):
        if not self.history:
            return
        action = self.history[-1]['action']
        data = self.history[-1]['data']
        self.future.append(self.history.pop())

        if action == self.ACTION_LOAD_ANNOTATION:
            filename = data['ori_filename']
            self.facade = filename
        elif action == self.ACTION_SET_OBJECT:
            index = data['index']
            obj = data['ori_value']
            self.facade.annotation.object[index] = deepcopy(obj)
            self._update_polygon(index)
            self.active_index = index
        elif action == self.ACTION_INSERT_OBJECT:
            index = data['index']
            self.poly_selector.delete_polygon(index)
            del self.facade.annotation.object[index]

    def redo(self):
        if self.future:
            self._do(**self.future.pop(), clear_future=False)

    def backup(self, path=None):
        """Save a backup version of the current annotation"""
        if path is None:
            path = self.facade.annotation_path + '~'
        self.facade.save_annotation(path=path)
        # pickle.dump(self.history, 'history.pkl')
        return path

    def save(self):
        self.facade.save_annotation()

    def get_label_box(self):
        return self._label_box

    def set_label_box(self, label_box):
        self._label_box = label_box

        ori_onselect = label_box.onselect

        def new_onselect(index):
            self._on_label_box_select(index)
            if ori_onselect is not None:
                ori_onselect(index)

        new_onselect.ori_onselect = ori_onselect

        label_box.onselect = new_onselect

    label_box = property(get_label_box, set_label_box)

    def _on_label_box_select(self, i):
        if i >= 0:
            if self.is_editing():
                self.label.set_text(self._label_box.names[i])
            else:
                self.filter_objects(name=self._label_box.names[i])
                self.label.set_text(self._label_box.names[i])
        else:
            # No parameters clears the filter
            if not self.is_editing():
                self.filter_objects()

    def create_label_box(self, ax: plt.Axes) -> LabelBoxes:
        label_box = LabelBoxes(ax, self.names, self.colors)
        return label_box

    def start_editing(self, x=0, y=0):
        """Start editing a polygon"""
        index = self.active_index
        self._poly_editor_index = index
        self.poly_selector.select_enabled = False

        # From here below, active_index is -1 (selection is disabled)

        if index < 0:
            xy = np.array([[x, y]])
        else:
            xy = self.poly_selector.polygons[index].get_xy()
            self.label.set_text(self.facade.annotation.object[index].name)

        self.poly_editor = PolygonEditor(xy, ax=self.ax,
                                         complete=index >= 0,
                                         on_update=self._on_polygon_edited)

    def is_editing(self):
        return self.poly_editor is not None

    def _on_polygon_edited(self, x, y):
        self.label.set_visible(True)
        self.label.set_x((np.min(x) + np.max(x)) / 2)
        self.label.set_y((np.min(y) + np.max(y)) / 2)

    # noinspection SpellCheckingInspection
    def _make_object(self, pt, name=None, deleted=0, verified=0, occluded='no',
                     attributes=None, hasparts=None, ispartof=None, date=None,
                     id_=None, username=None):
        if name is None:
            name = self.active_label

        if username is None:
            username = self.username

        if id_ is None:
            id_ = uuid1().hex

        if date is None:
            date = datetime.now().strftime('%d-%h-%Y %H:%M:%S')

        if not isinstance(pt[0], dict):
            pt = [dict(x=x, y=y) for (x, y) in pt]

        result = EasyDict(name=name, deleted=deleted,
                          verified=verified,
                          occluded=occluded,
                          attributes=attributes,
                          date=date,
                          id=id_,
                          parts=EasyDict(hasparts=hasparts, ispartof=ispartof),
                          polygon=EasyDict(username=username, pt=pt))
        return result

    def finish_editing(self, cancel=False):
        if not self.is_editing():
            return

        index = self._poly_editor_index

        if not cancel:
            # Set the current polygon based on the editor
            xy = np.column_stack([self.poly_editor.x, self.poly_editor.y])
            new_object = self._make_object(xy)

            if index < 0:
                # We were not editing an existing polygon
                index = len(self.poly_selector.polygons)

                self._do(self.ACTION_INSERT_OBJECT,
                         dict(index=index, value=new_object))
            else:
                # Modify select fields of the original object
                ori_object = self.facade.annotation.object[index]
                proto = deepcopy(ori_object)
                proto.pop('name')  # Replace it with the active label
                proto.pop('polygon')  # Replace it with xy from the editor
                proto.pop('date')  # Replace it with datetime.now
                new_object.update(proto)

                self._do(self.ACTION_SET_OBJECT,
                         dict(index=index, value=new_object, ori_value=ori_object))

            # Aggressively save our work
            self.backup()

        # Get rid of the editor
        self.poly_editor.remove()
        self.poly_editor = None

        # Re-enable the selector and restore the active index
        self.poly_selector.select_enabled = True
        self.active_index = index
        self._on_poly_activate(index)

    def filter_objects(self, name=None, username=None, verified=None, visible=None, expr=None,
                       hide=True):
        """ Prevent certain objects from being selectable
        
        Parameters:
        - name: Selectable objects must have the same name
        - username: Selectable objects must have the same username
        - verified: Selectable objects must have the same 'verified' status
        - visible: Selectable polygons must have the same visibility
        - expr:  A lambda expression (or function) that is true if an object should be selectable.
        - hide: Whether to hide unselectable objects. 
        """
        for i, o in enumerate(self.facade.annotation.object):
            selectable = not o.deleted
            if name is not None:
                selectable &= o.name == name
            if username is not None:
                selectable &= o.polygon.username == username
            if verified is not None:
                selectable &= bool(int(o.verified)) == verified
            if expr is not None:
                selectable &= expr(o)
            if visible is not None:
                selectable &= self.poly_selector.polygons[i].get_visible() == visible

            self.poly_selector.selectable[i] = selectable

            if hide:
                self.poly_selector.polygons[i].set_visible(selectable)

        # Make sure the currently selected object is not filtered out
        if not self.poly_selector.selectable[self.poly_selector.active_index]:
            self.poly_selector.active_index = -1

        # Make sure the current hover object is not filtered out
        if not self.poly_selector.selectable[self.poly_selector.hover_index]:
            self.poly_selector.hover_index = -1

        # (I cannot remember if unselectable objects are drawn differently)
        self.poly_selector.update()

        # Save the arguments so we can re-apply the filter if we load the next facade
        self.last_filter = dict(name=name, username=username, verified=verified,
                                visible=visible, expr=expr, hide=hide)

    def _update_polygon(self, index):
        """Update the polygon associated with an object"""
        o = self.get_object(index)
        p = self.poly_selector.polygons[index]
        xy = self.facade.rectified(o)
        p.set_xy(xy)
        label = self.z_orders[o.name]
        p.set_facecolor(self.colors[label].tolist() + [self.fill_alpha])
        p.set_edgecolor(self.colors[label])

        # If the label has changed, update the color

        if o.deleted:
            p.set_visible(False)
            self.poly_selector.selectable[index] = False
            if self.active_index == index:
                self.active_index = -1
            if self.hover_index == index:
                self.hover_index = -1
        else:
            # TODO: Save selectable state in history???
            p.set_visible(True)
            self.poly_selector.selectable[index] = True

        self.poly_selector.refresh_polygon(index)

        # We manage the label -- make sure it is positioned over the (updated) poly
        self.label.set_text(o.name)
        self.label.set_x((xy[:, 0].min() + xy[:, 0].max()) / 2)
        self.label.set_y((xy[:, 1].min() + xy[:, 1].max()) / 2)

        return p

    def _make_polygon(self, o):
        assert o is not None

        label = self.z_orders[o.name]
        props = dict(zorder=self.facade.z_orders[o.name],
                     facecolor=self.colors[label].tolist() + [self.fill_alpha],
                     edgecolor=self.colors[label])

        p = Polygon(self.facade.unrectified(o), **props)
        return p

    def duplicate_active(self):
        # Duplicating should commit the current edit...
        if self.is_editing():
            self.finish_editing()

        if self.active_object is None:
            return

        self.insert_object(self.active_index + 1, deepcopy(self.active_object))

    def delete_active(self, deleted=1):
        if self.is_editing():
            self.finish_editing()

        if self.active_object is None:
            self.canvas.toolbar.set_message('No selection -- cannot delete')
            return

        obj = deepcopy(self.active_object)
        obj.update(deleted=deleted)
        self.set_object(self.active_index, obj)

    def load_next_annotation(self, direction=+1):
        self.save()
        self._annotation_index = (self._annotation_index + direction) % len(self._annotation_list)
        self.load_annotation(self._annotation_list[self._annotation_index])
        if self.last_filter is not None:
            self.filter_objects(**self.last_filter)

    def load_previous_annotation(self, direction=-1):
        self.load_next_annotation(direction)

    def update(self):
        self.canvas.draw_idle()

    def _on_poly_activate(self, index):
        if self.on_select is not None:
            self.on_select(index)

        if index < 0:
            self.label.set_visible(False)
        else:
            bb = self.poly_selector.polygons[index].get_extents()
            bb = bb.transformed(self.ax.transData.inverted())
            x, y = bb.corners().mean(0)
            self.label.set_x(x)
            self.label.set_y(y)
            self.label.set_zorder(len(self.poly_selector.polygons) + 1)
            self.label.set_text(self._facade.annotation.object[index].name)
            self.label.set_visible(True)

    def _on_poly_hover(self, index):
        if self.on_hover is not None:
            self.on_hover(index)

    def _on_key_press(self, event: KeyEvent):
        if event.inaxes is not self.ax:
            return

        if event.key == ' ':
            if not self.is_editing():
                self.start_editing(event.xdata, event.ydata)
            else:
                self.finish_editing()
        elif event.key == 'escape':
            if self.is_editing():
                self.finish_editing(cancel=True)
        elif event.key == '*':
            # Toggle visibility
            for (p, s) in zip(self.poly_selector.polygons, self.poly_selector.selectable):
                if s:
                    p.set_visible(not p.get_visible())
            self.update()
        elif event.key == '5':
            if self.active_object:
                self.poly_selector.fit_active()
        elif event.key == 'ctrl+z':
            if not self.is_editing():
                self.undo()
        elif event.key == 'ctrl+Z':
            if not self.is_editing():
                self.redo()
        elif event.key == 's':
            self.save()
            self.canvas.toolbar.set_message("saved")
        elif event.key == 'd':
            self.finish_editing()
            self.duplicate_active()
            self.start_editing()
        elif event.key == 'e':
            if not self.is_editing():
                self.delete_active()
        elif event.key == 'shift+pageup':
            if not self.is_editing():
                self.load_next_annotation()
                self.poly_selector.select_next()
        elif event.key == 'shift+pagedown':
            if not self.is_editing():
                self.load_previous_annotation()
                self.poly_selector.select_prev()
        elif event.key == '\\':
            if self.get_active_object() is not None:
                o = deepcopy(self.get_active_object())
                o.name = 'unlabeled'
                self.set_object(self.active_index, o)
        elif event.key == ']':
            self.poly_selector.select_next()
            while self.get_active_object() is None:
                self.load_next_annotation()
                self.poly_selector.fit_all()
                self.poly_selector.select_next()
                self.canvas.draw()
        elif event.key == '[':
            self.poly_selector.select_prev()
            while self.get_active_object() is None:
                self.load_previous_annotation()
                self.poly_selector.select_prev()

    def connect(self, event, callback):
        self.cids.append(self.canvas.mpl_connect(event, callback))

    def connect_events(self):
        self.connect('key_press_event', self._on_key_press)

    def disconnect_events(self):
        while self.cids:
            self.canvas.mpl_disconnect(self.cids.pop())


def run():
    from argparse import ArgumentParser

    # Parse arguments -- read the help below or run with '-h' for help.
    p = ArgumentParser(description="Interactive tool to visualize, and edit, annotations")
    p.add_argument('--input', '-i', type=str, nargs='*',
                   help="The files to edit. If none are specified, we "
                        "move through them all from oldest to newest")
    p.add_argument('--root', type=str, default=DATA_ROOT,
                   help="The root of the dataset.")
    p.add_argument('--folder', type=str, default='merged',
                   help="The folder (grooup of annotations) to work with")
    p.add_argument('--username', '-u', type=str,
                   help='name of the person editing annotations')
    p.add_argument('--auto-advance', type=bool,
                   help="Automatically advance to the next file when no more selectable items found")
    args = p.parse_args()

    # This code is designed to work insided a Jupyter notebook
    # so we use a matplotlib Figure as our GUI
    fig = plt.figure(figsize=(9, 6), frameon=False)

    # Prevent the toobar from handling some key press events (e.g. s)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

    splits = [0.09, 1]
    lbax = plt.axes([0, 0.0, 1, splits[0]], xticks=[], yticks=[])
    ax = plt.axes([0, splits[0], 1, splits[1] - splits[0]],
                  xticks=[], yticks=[])
    ax.axis('equal')

    if args.input:
        annotations = args.input
    else:
        annotations = glob(os.path.join(args.root, 'Annotations', args.folder, '*.xml'))
        annotations = sorted(annotations, key=lambda x: os.stat(x).st_mtime)

    facade = fsi.FacadeSubImage(annotation=annotations[0], root=args.root)
    ae = AnnotationEditor(ax,
                          facade=facade,
                          root=args.root,
                          username=args.username,
                          folder=args.folder,
                          annotations=annotations
                          )
    ae.set_label_box(ae.create_label_box(lbax))


    # TODO: Can this be made part of AnnotationEditor?
    def on_resize(event: ResizeEvent):
        h = ae.get_label_box().get_preferred_height()
        box = ae.ax.get_position()
        new_label_box = Bbox.from_bounds(box.xmin, 0, box.width, h)
        ae.get_label_box().ax.set_position(new_label_box)

        # Resize the annotation editor itself
        box = ae.ax.get_position()
        new_editor_box = Bbox.from_bounds(box.xmin,
                                          h,
                                          box.width,
                                          1-h)
        ae.ax.set_position(new_editor_box)

    resize_cid = fig.canvas.mpl_connect('resize_event', on_resize)
    plt.show()
    return ae


# python annnotation_editor.py
if __name__ == '__main__':
    plt.ioff()
    ae = run()