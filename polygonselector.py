"""A matplotlib widget for selecting polygons.
author: femianjc@miamioh.edu

NOTE: This does not extend matplotlib.widget.Widget because I did not have time
      to learn the ins and outs of the base class. Making it extend widget 
      probably wont be hard. 
"""


import matplotlib.patches
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.text import Text

import numpy as np


class PolygonSelector(object):
    """A Polygon Selection Tool
    
    Renders and allows selection of polygons. 

    
    - Hovering over an item highlights it
    - <kbd>LMB</kbd> to select an item
    - <kbd>RMB</kbd> to start dragging / panning the view
    - <kbd>CTRL</kbd> <kbd>wheel</kbd> to zoom in and out around the mouse cursor
    - <kbd>p</kbd> starts pan mode 
    - <kbd>+</kbd> zooms in to cursor, and <kbd>-</kbd> zooms out
    - <kbd>4</kbd>,<kbd>8</kbd>,<kbd>6</kbd>,<kbd>2</kbd> pan lft, up, right, down. 
    - <kbd>PgUp</kbd> and <kbd>PgDn</kbd> zoom to the next or previous label. 
    
    The semantics are this:
    - We own a list of polygons, created using our 'add_polygon' method. 
    - We maintain a list of our polygons as `self.polygons`.  You may use
      `self.poygons.index` to find the index of a polygon.
    - As the mouse moves, we predict which polygon would be selected if the 
      mouse cicked and we render that as a `hover` polygon. 
    - When the mouse is clicked, the polygon just _under_ the currently selected
      polygon is chosen (based on zorder). If there is no polygon under the 
      active polygon then the topmost polygon is selected.
    - When we "zoom" to a polygon by pressing <kbd>PgUp</kbd>, <kbd>SPACEBAR</kbd>, 
      or <kbd>PgDn</kbd> then the polygon is selected, centered in the view, 
      and the view is set to be just slightly larger than the polygon. 
      
    """

    def __init__(self, ax, onactivate=None, onhover=None, expandby=1.5):
        """
        Parameters:
        
        - ax: The axis to render onto. 
        - onactivate: A callback, called with the index of a polygon 
          just before it is set as the active polygon. 
        - onhover: A callback, called with the index of a polygon just 
          before it is set as the hover (highlighted) polygon. 
                   
        """
        self.ax = ax
        self.canvas = self.ax.figure.canvas
        self.polygons = []
        self.selectable = []
        self.argsorted = None
        self.onactivate = onactivate
        self.onhover = onhover
        self.expandby = expandby
        
        self._select_enabled = True
        
        props = dict(visible=False, facecolor=(1,1,0, 0.4), edgecolor=(1,1,1), linewidth=2)
        self.hover_highlighter = self.ax.add_patch(Polygon([[0,0]], **props))
        
        props = dict(visible=False, facecolor=(0,0,1, 0.4), edgecolor=(0,0,1), linewidth=2)
        self.active_highlighter = self.ax.add_patch(Polygon([[0,0]], **props))

        self._hover_index = -1
        self._active_index = -1  # Never touch this -- use the self.active_index property instead
        
        # Save the button-press event to handle dragging
        self._event_on_press = None
        
        self.scroll_speed = 0.1
        
        self.zooming = False
        
        self.cids = []
        self.connect_events()
        
    def zsort(self):
        self.argsorted = np.argsort([p.get_zorder() for p in self.polygons])
        
    def update(self):     
        # Redraw
        self.canvas.draw_idle()
        
    @property
    def select_enabled(self):
        return self._select_enabled
    
    @select_enabled.setter
    def select_enabled(self, value):
        if value != self._select_enabled:
            self._select_enabled = value
            
            if self._select_enabled == False:
                self.active_index = -1
                self.hover_index = -1
    
    @property
    def active_index(self):
        """The index of the active polygon in `self.polygons`"""
        return self._active_index
    
    @active_index.setter
    def active_index(self, value):
        if self.select_enabled == False:
            value = -1
            
        if value != self.active_index:
            self._active_index = value
            
            self.refresh_active_highlighter()
        
            if self.onactivate is not None:
                self.onactivate(value)
       

    def refresh_active_highlighter(self):
        """Update the highlighter for the active object"""
        if self.active_polygon:
            self.active_highlighter.set_visible(True)
            self.active_highlighter.set_xy(self.active_polygon.get_xy())
            self.active_highlighter.set_zorder(self.active_highlighter.get_zorder() + 1)
        else:
            self.active_highlighter.set_visible(False)
        
            
    @property
    def hover_index(self):
        """The index of the hover polygon in `self.polygons`"""
        return self._hover_index
    
    @hover_index.setter
    def hover_index(self, value):
        if self.select_enabled == False:
            value = -1
            
        if value != self._hover_index:
            self._hover_index = value
            
            self.refresh_hover_highlighter()

            if self.onhover is not None:
                self.onhover(value)

    def refresh_hover_highlighter(self):
        """Update the highlighter for the hover object"""
        if self.hover_polygon:
            self.hover_highlighter.set_visible(True)
            self.hover_highlighter.set_xy(self.hover_polygon.get_xy())
            self.hover_highlighter.set_zorder(self.hover_polygon.get_zorder() + 1)
        else:
            self.hover_highlighter.set_visible(False)
        self.update()
        
        
    def add_polygon(self, xy, selectable=True, **kwargs) -> matplotlib.patches.Polygon:
        self.insert_polygon(len(self.polygons), xy, selectable, **kwargs)
    
    def delete_polygon(self, index):
        p = self.polygons[index]
        p = p.remove()
        del self.polygons[index]
        del self.selectable[index]
        self.argsorted = None
        self.canvas.draw_idle()
        return p
    
    def insert_polygon(self, index, xy, selectable=True, **kwargs):
        if isinstance(xy, Polygon):
            poly = xy
        else:
            poly = Polygon(xy, **kwargs)
        poly = self.ax.add_patch(poly)
        self.polygons.insert(index, poly)
        self.selectable.insert(index, selectable)
        self.argsorted = None
        self.canvas.draw_idle()
        return poly      
    
    def refresh_polygon(self, index):
        if index == self.hover_index:
            self.refresh_hover_highlighter()
        if index == self.active_index:
            self.refresh_active_highlighter()
        self.update()
    
    @property
    def hover_polygon(self):
        if self.hover_index >= 0:
            return self.polygons[self.hover_index]
        else:
            return None
    
    @property 
    def active_polygon(self):
        if self.active_index >= 0:
            return self.polygons[self.active_index]
        else:
            return None
    
    def connect(self, event, handler):
        self.cids.append(self.canvas.mpl_connect(event, handler))
        
    def connect_events(self):
        self.connect('motion_notify_event', self._on_motion)
        self.connect('scroll_event', self._on_scroll)
        self.connect('key_press_event', self._on_key_press)
        self.connect('key_release_event', self._on_key_release)
        self.connect('button_press_event', self._on_button_press)
        self.connect('button_release_event', self._on_button_release)
        
    
    def disconnect_events(self):
        for cid in self.cids:
            self.canvas.mpl_disconnect(cid)
        self.cids = []
        
    def clear(self):
        self.active_index = -1
        self.hover_index = -1
        for p in self.polygons:
            p.remove()
        del self.polygons[:]
        del self.selectable[:]
        self.update()
        
    def find_polygon(self, xy, selectable=True):
        # Start just below the active index (so we can select overlapping polygons)
        if self.argsorted is None:
            self.zsort()
            
        if self.active_index >= 0:
            ai = np.where(self.argsorted == self.active_index)[0].item()
        else:
            ai = 0
        n = len(self.polygons)
        for j in range(n):
            i = (ai-j-1)%n
            if selectable and not self.selectable[self.argsorted[i]]:
                continue
            if self.polygons[self.argsorted[i]].contains_point(xy):
                return self.argsorted[i]        
        return -1
    
    def select_at(self, xy):
        if not self.select_enabled:
            return
        self.active_index = self.find_polygon(xy)
        self.hover_index = self.find_polygon(xy)
    
    def _on_button_press(self, event):
        if event.inaxes != self.ax: return
        
        self._event_on_press = event
        
        if event.dblclick or (event.button == 1):
            self.select_at((event.x, event.y))
        elif event.button == 3:
            # Pretend we are pressing the LMB (1)
            self.ax.start_pan(event.x, event.y, 1)

    
    def _on_button_release(self, event):
        self._event_on_press = None
        
        if event.button == 3:
            if self.ax.__dict__.get('_pan_start') is not None:
                self.ax.end_pan()
    
    def zoom(self,amount=1,  xy=None):
        
        amount = 2**(amount)
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        if xy is None:
            x = (xmin + xmax)/2.
            y = (ymin + ymax)/2.
        else:
            x, y = xy
        xmin = x + (xmin-x)*amount
        ymin = y + (ymin-y)*amount
        xmax = x + (xmax-x)*amount
        ymax = y + (ymax-y)*amount
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)        
    
    def _on_scroll(self, event):
        if self.zooming:
            x, y = event.xdata, event.ydata
            self.zoom( event.step*self.scroll_speed, (x, y))
            self.update()
        

    def _on_motion(self, event):
        if event.inaxes != self.ax: return
        
        if event.button == None:
            # Update the hover index
            if self.select_enabled:
                index = self.find_polygon((event.x, event.y))
                if index != self.hover_index:
                    self.hover_index = index
        elif event.button == 3: #RMB
            # Dragging (pretend we are pressing LMB in pan mode)
            if self.ax.__dict__.get('_pan_start') is not None:
                self.ax.drag_pan(1, event.key, event.x, event.y)
                self.update()
            
            
    def pan(self, dx, dy, units='percent'):
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        
        if units == 'percent':
            dx *= (xmax-xmin)/100.
            dy *= (ymax-ymin)/100.
        
        xmin += dx
        xmax += dx
        ymin += dy
        ymax += dy
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.update()
        
    def pan_to_active(self):
        if self.active_polygon is None:
            return
        vec = self.active_polygon.get_xy().mean(0)-self.ax.viewLim.corners().mean(0)
        bbox = self.ax.viewLim.translated(*vec)
        self.ax.set_xlim(bbox.x0, bbox.x1)
        self.ax.set_ylim(bbox.y0, bbox.y1)
        self.update()
 
    def pan_to_point(self, x, y):
        vec = np.array(x, y)-self.ax.viewLim.corners().mean(0)
        bbox = self.ax.viewLim.translated(*vec)
        self.ax.set_xlim(bbox.x0, bbox.x1)
        self.ax.set_ylim(bbox.y0, bbox.y1)
        self.update()
 
    def fit_active(self, expand=None):
        if self.active_polygon is None:
            self.ax.set_xbound(*self.ax.dataLim.intervalx)
            self.ax.set_ybound(*self.ax.dataLim.intervaly)
            return 
        
        if expand is None:
            expand = self.expandby
            
        bb = self.active_polygon.get_extents()
        bb = bb.transformed(self.ax.transData.inverted())
        bb = bb.expanded(expand, expand)
        self.ax.viewLim.set(bb)
        self.update()
    
    def select_next(self, selectable=True, direction=1):
        if not self.select_enabled:
            return
        
        i = (self.active_index + direction) % len(self.polygons)
        if selectable:
            found = False
            for j in range(len(self.polygons)):
                if not self.selectable[i]:
                    i = (i+direction) % len(self.polygons)
                else:
                    found = True
                    break
        if found:   
            self.active_index = i
        else:
            self.active_index = -1
        self.update()

    def select_prev(self, selectable=True, direction=-1):
        self.select_next(selectable, direction)

    def fit_next(self, expand=None, selectable=True):
        if not self.select_enabled:
            return
        if expand is None:
            expand = self.expandby
        self.select_next(selectable)
        self.fit_active(expand)        
        
    def fit_previous(self, expand=None, selectable=True):
        if not self.select_enabled:
            return
        
        if expand is None:
            expand = self.expandby
        self.select_prev(selectable)
        self.fit_active(expand)        
        
    
    def _on_key_press(self, event):
        if event.inaxes != self.ax: return

        self.canvas.toolbar.set_message(f'pressed {event.key}')
        #print(event.key)
        if event.key == 'p':
            self.canvas.toolbar.pan()
        elif event.key == 'escape':
            self.active_index = -1
            self.update()
        elif event.key == 'control': 
            self.zooming = True  # event.key not set when scrolling, this is a hack
        elif event.key == '-':
            self.zoom(1, (event.xdata, event.ydata))
            self.update()
        elif event.key == '+':
            self.zoom(-1, (event.xdata, event.ydata))
            self.update()
        elif event.key == '4':
            self.pan(-10, 0)
        elif event.key == '6':
            self.pan(10, 0)
        elif event.key == '2':
            self.pan(0, -10)
        elif event.key == '8':
            self.pan(0, 10)
        elif event.key=='pageup':
            self.fit_next()
        elif event.key == 'pagedown':
            self.fit_previous()
        elif event.key == 'home':
            self.ax.set_xbound(*self.ax.dataLim.intervalx)
            self.ax.set_ybound(*self.ax.dataLim.intervaly)
            
    def _on_key_release(self, event):
        if event.inaxes != self.ax: return
        
        if event.key == 'control':
            self.zooming = False
        
