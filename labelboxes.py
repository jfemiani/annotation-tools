import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox

class LabelBoxes(object):
    """A Label Selection Widget. 
    
    This displays a list of labels, each in a different color, and allows
    them to be selected. 
    
    The semantics are:
    - a _hover_ label is rendered under the mouse to indicate which label
      would be selected if the mouse clicked. 
    - a _active_ label is rendered to indicate the currently selected label.
    
    A callback `onselect(index)` is invoked whenever the active label changes. 
    
    """
    def __init__(self, ax, names, colors, onselect=None, pad=3, margin=5): 
        """
        Parameters:
        - *ax:* The axist to render onto. 
        - *labels:* A list of label names (strings)
        - *colors:* A list of label colorts (for matplotlib)
        - *onselect:* A callback invoked with the index of the active label. 
        - *pad:* The amount of padding to use in the boxes around each label (in pixels)
        - *margin:* The amount to indent the labels within the widget (in pixels)
        """
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.labels = []
        self.rows = []
        self.colors = colors
        self.names = names
        self.pad = pad 
        self.margin = margin
        self.onselect= onselect
        
        self._hover_index = -1
        self._active_index = -1
        
        self.eventson = True # Same ideas as matplotlib.widgets.Widget
        self.drawon = True
        
        props = dict(fill=False, edgecolor='y', linewidth=3, joinstyle='round', alpha=0.5, )
        self.hoverbox = self.ax.add_patch(Rectangle((0,0), 0,0, **props,
                                                    zorder=len(names), visible=False))
        
        props = dict(fill=False, edgecolor='b', linewidth=3, joinstyle='round', alpha=0.5)
        self.activebox = self.ax.add_patch(Rectangle((0,0), 0,0, **props, 
                                                     zorder=len(names)+1, visible=False))
        
        
        self.cids = []
        
        self._place_labels(names, colors)
        self.connect_events()
    
    def ignore(self, event):
        """ Return true if an event should be ignored 
        -- call an the beginning of each event handler"""
        # Pattern is to sequentially test conditions 
        # and return true if any condition is true
        
        if self.ax is not event.inaxes:
            return True # Also handles event.ax == None...
        
        if not self.eventson:
            return True
        return False
    
    @property
    def active_index(self):
        return self._active_index
    
    @active_index.setter
    def active_index(self, value):
        if value != self._active_index:
            self._active_index = value
            if self.onselect: 
                self.onselect(value)
                
            # Highlight currently selected label
            todata = self.ax.transData.inverted()
            if self.active_index >= 0:
                bb = self.labels[self.active_index].get_bbox_patch()
                self.activebox.set_bounds(*bb.get_extents().transformed(todata).bounds)
                self.activebox.set_visible(True)
            else:
                self.activebox.set_visible(False)
        
            self.update()
            
    @property 
    def hover_index(self):
        return self._hover_index

    @hover_index.setter
    def hover_index(self, value):
        if value != self._hover_index:
            self._hover_index = value
            # TODO: self.onhover

            # Highlight the label that we are hovering over
            todata = self.ax.transData.inverted()
            if self.hover_index >= 0:
                bb = self.labels[self.hover_index].get_bbox_patch()
                self.hoverbox.set_bounds(*bb.get_extents().transformed(todata).bounds)
                self.hoverbox.set_visible(True)
            else:
                self.hoverbox.set_visible(False)
            
            self.update()
            
        
    def connect(self, event, callback):
        self.cids.append(self.canvas.mpl_connect(event, callback))
        
    def connect_events(self):
        self.connect('button_press_event', self._on_button_press)
        self.connect('motion_notify_event', self._on_motion)
        self.connect('resize_event', self._on_resize)

    def _on_resize(self, event):
        self._place_labels(self.names, self.colors)

    def _on_button_press(self, event):
        if self.ignore(event): return
            
        index = self.pick_label(event)
        if index == self.active_index:
            self.active_index = -1
        else:
            self.active_index = index
        
        self.update()
            
    def _on_motion(self, event):
        if self.ignore(event): return
        
        todata = self.ax.transData.inverted()
        self.hover_index = self.pick_label(event)
       
        self.update()
        
    def get_label(self, index):
        return self.labels[index].get_text()
        
    @property
    def active_label(self):
        return self.get_label(self.active_index)
    
    def pick_label(self, event):
        for index, label in enumerate(self.labels):
            bbp = label.get_bbox_patch()
            if bbp.contains_point((event.x, event.y)):
                return index
        return -1
        
    def update(self):   
        if self.drawon:
            self.canvas.draw_idle()
        
    def _get_labels_bbox(self):
        # Start with computing the union of all label boxes
        todata = self.ax.transData.inverted()
        renderer = self.canvas.get_renderer()
        boxes = []
        for label in self.labels:
            # Update box for new label position
            label.draw(renderer)        
            boxes.append(label.get_bbox_patch().get_window_extent(renderer).transformed(todata))
        bbox = Bbox.union(boxes)
        return bbox

    def get_preferred_height(self):
        data_bbox = self._get_labels_bbox()
        display_bbox = data_bbox.transformed(self.ax.transData)
        figure_bbox = display_bbox.transformed(self.ax.figure.transFigure.inverted())
        return figure_bbox.height
        
    def _place_labels(self, labels, colors):
        for label in self.labels:
            label.remove()
        del self.labels[:]

        self.rows = [[]]
        renderer = self.canvas.get_renderer()
        todata = self.ax.transData.inverted()
        xmargin, ymargin = todata.transform((self.margin, self.margin))
        x, y = xmargin, ymargin
        
        # Let the labels flow from left to right, wrapping around
        for label_index in range(len(labels)):
            s = labels[label_index]
            
            label = Text(x, y, s, verticalalignment='top',  
                         bbox=dict(facecolor=colors[label_index], edgecolor='w', pad=self.pad))
            
            # I could not see the dark text on some backgrounds
            label.set_path_effects([
                PathEffects.withSimplePatchShadow(offset=(1,-1), shadow_rgbFace=(1., 1.,0.5)),
                ])
            label = self.ax.add_artist(label)
            
            # Keep both a master list of labels, and a list of labelson each row
            self.labels.append(label)
            self.rows[-1].append(label)
            
            # I have to render the label once in order to determine its actual size
            label.draw(renderer)
            bbox = label.get_bbox_patch().get_window_extent(renderer).transformed(todata)
            if (x + bbox.width) > 1:
                x = xmargin
                y -= bbox.height
                label.set_position((x, y))
                self.rows.append([self.rows[-1].pop()])
            x +=  bbox.width
            
        # After some tinkering, I am centering the labels in the axis 
        bbox = self._get_labels_bbox()
        
        labels_center = bbox.corners().mean(0)
        view_center = self.ax.viewLim.corners().mean(0)
        for label in self.labels:
            x, y = label.get_position() + (view_center - labels_center)
            label.set_position((x, y))
        
