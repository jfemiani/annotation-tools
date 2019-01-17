import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets
import facadesubimage as fsi
from matplotlib.patches import Polygon
from easydict import EasyDict
from datetime import datetime


class LabelExplorer(matplotlib.widgets.Widget):
    """ Visualize (and to some extent modify) labels. 
    
    This uses python's main plotting tool (matplotlib) to plot the labels.
    
    Matplotlib has support for some GUI-neutral interaction, 
    which means that the same code can be run from within a Jupyter Notebook plot
    or in a window on any operating system that supports maptlotlib. 
    
    We use this to provide some polygon editing abilities. 
    """
    selector: matplotlib.widgets.PolygonSelector

    def __init__(self, ax, xmls, iou=0.3, username='femiani', pad=0.5, startat=0):
        """
        Parameters:
        - ax:       The main matplotlib axis to render into. 
        - xmls:     A list of annotation XML files to iterate over. 
        - username: If an annotation is added, this is who we mark as the annotator
        - pad:      When we try to zoom in on a set of objects, this is the percent of 
                    extra spece we add around the edges. 
        - startat:  In case we are interupted, which XML should we resume inspecting. 
        """
        super().__init__(ax)
        self.iou = iou
        self.duplicates = []
        self.duplicate_index = -1
        self.ax = ax
        self.fig = ax.figure
        self.xmls = xmls
        self.username = username
        self.pad = pad

        # The index of the curren XML file.
        self.annotation_index = startat

        # Load the XML file 
        self._next_annotation(self.annotation_index)

        # Update the plot 
        self.objects = self._plot_dup()

        # The figure manager uses some keys -- most notably the arrow keys. 
        # We want to use those keys for something else (e.g. moving the polygon)
        # So we will disconnect the manager and forward keystrokes that we have not 
        # handles in our own 'key_press_event' handler. 
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)

    def onselect(self, pts):
        pts = np.array(pts)
        self.poly.set_xy(pts)

    def _next_annotation(self, index=None):
        """ Load the annotation specified (default: self.annotation_index+1)
        
        Parameters:
        - index: The index of the next annotation to load (default: self.annotation_index+1)
        """
        if index is None:
            self.annotation_index += 1
        else:
            self.annotation_index = index

        self.annotation = fsi.FacadeSubImage(annotation=self.xmls[self.annotation_index])

    # noinspection PyProtectedMember
    def _set_poly(self, ob):
        """Set our polygon editor widget to the shape of 'ob'
        """
        pts = self.annotation.unrectified(ob.polygon.pt)

        self.selector._xs[:] = pts[:, 0].tolist() + [pts[-1, 0]]
        self.selector._ys[:] = pts[:, 1].tolist() + [pts[-1, 1]]
        self.selector._polygon_completed = True
        self.selector._draw_polygon()
        self.selector.onselect(self.selector.verts)
        self.selector.update()

    def _plot_dup(self):
        ax = self.ax
        a = self.annotation
        d = self.duplicates
        index = self.duplicate_index

        ax.cla()
        ax.imshow(self.annotation.image, interpolation='bicubic')

        if self.duplicate_index < len(self.duplicates):
            objects = [a.annotation.object[d[index][0]], a.annotation.object[d[index][1]]]
            bbox = a.bbox(objects, pad=self.pad)
            a.plot(ax, objects=objects, alpha=0.2)

            # Keep the plot square
            cx, cy = (bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.,
            sc = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2.
            ax.set_xlim(cx - sc, cx + sc)
            ax.set_ylim(cy + sc, cy - sc)

            self.poly = Polygon(a.rectified(objects[0]), color='b', alpha=0.3, zorder=50)
            self.poly = ax.add_patch(self.poly)
            self.selector = matplotlib.widgets.PolygonSelector(ax, useblit=True, onselect=self.onselect)
            self._set_poly(objects[0])
        else:
            self.poly = None
            self.selector = None
            a.plot(ax, alpha=0.2)
            objects = []

        ax.set_title(f"({index} of {len(d)}) in xml ({self.annotation_index} of {len(self.xmls)})")

        return objects

    # noinspection PyProtectedMember
    def onkeypress(self, event):
        a = self.annotation

        if event.key == "enter":
            # Save the current polygon as a new object
            if self.poly:
                newobject = EasyDict(self.objects[0])
                newobject.polygon.pt = [EasyDict(x=x, y=y) for (x, y) in self.poly.xy]
                newobject.date = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
                newobject.polygon.username = self.username
                a.annotation.object.append(newobject)

            # Delete the two original objects (or mark them to delete)
            for o in self.objects:
                o.deleted = 1

            # Advance to the next duplicate
            self.duplicate_index += 1

            # If we commit a file, save it and advance to the next one.
            if self.duplicate_index >= len(self.duplicates):
                a.remove_deleted()
                a.save_annotation()
                self._next_annotation()
                while len(self.duplicates) == 0:
                    self._next_annotation()

            # Render the current duplicate and update our estimated polygon
            self.objects = self._plot_dup()
        elif event.key == ' ':
            if len(self.objects):
                self.objects = self.objects[1:] + self.objects[:1]
                self._set_poly(self.objects[0])
        elif event.key == 'left':
            self.selector._xs[:] = [x - 1 for x in self.selector._xs]
            self.selector._draw_polygon()
            self.selector.onselect(self.selector.verts)
            self.selector.update()
        elif event.key == 'right':
            self.selector._xs[:] = [x + 1 for x in self.selector._xs]
            self.selector._draw_polygon()
            self.selector.onselect(self.selector.verts)
            self.selector.update()
        elif event.key == 'up':
            self.selector._ys[:] = [y - 1 for y in self.selector._ys]
            self.selector._draw_polygon()
            self.selector.onselect(self.selector.verts)
            self.selector.update()
        elif event.key == 'down':
            self.selector._ys[:] = [y + 1 for y in self.selector._ys]
            self.selector._draw_polygon()
            self.selector.onselect(self.selector.verts)
            self.selector.update()
        elif event.key == 'c':
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            cx, cy = self.poly.xy.mean(0)
            w = x1 - x0
            h = y1 - y0
            self.ax.set_xlim(cx - w / 2., cx + w / 2.)
            self.ax.set_ylim(cy - h / 2., cy + h / 2.)
            self.fig.canvas.draw()
        else:
            self.fig.canvas.manager.onkeypress(event)

    # noinspection SpellCheckingInspection
    @staticmethod
    def run():
        from argparse import ArgumentParser
        p = ArgumentParser()
        p.add_argument('--startat', type=int, default=0)
        p.add_argument('--hasdups', type=str, nargs='*',
                       help='A list of  XML annotations which may have duplicate objects')
        p.add_argument('--iou', type=float, default=0.3,
                       help="Them max IoU allowed between non-duplicated objects. "
                            "Lower values will return more duplicates")
        p.add_argument('--username', type=str, default='anonymous',
                       help='The username of the the annotator (you). Will'
                            'be used as the username of new objects')
        p.add_argument('--pad', type=float, default=0.5,
                       help="The size of the context to show around each object "
                            "when we zoom to an object. E.g. 0.5 means the box "
                            "is extended by 50% of the long edge")
        args = p.parse_args()

        fig = plt.figure(figsize=(8, 8))
        ax = plt.gca()
        fixer = LabelExplorer(ax,
                              xmls=args.hasdups,
                              startat=args.startat,
                              iou=args.iou,
                              username=args.username,
                              pad=args.pad)
        plt.ioff()
        fig.show()


if __name__ == '__main__':
    LabelExplorer.run()
