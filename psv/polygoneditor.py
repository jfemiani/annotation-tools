import numpy as np
from numpy import array
from copy import copy, deepcopy

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backend_bases import MouseEvent, KeyEvent


class PolygonEditor(object):
    """ Edit a polygon
    
    Creates a polygon tool that can be used to create and/or edit a polygon. 
    
    For an open polygon:
    - lmb:  Add a point to the end. Click the first point to close the polygon
    - ctrl+lmb: Select or drag a vertex or edge
    - shift+lmb: Drag the entire polygon
    
    For a complete polygon:
    - lmb: Select and drag a vertex or edge. If none are active, activates every point. 
    - x/y + lmb: Constrain motion to the x/y axis
    - n + lmb: Slide an edge keeping all lines parallel to their original direction
    - w: Split the active (or last) edge at a point (at the cursor)
    - e: Erase/collapse the active (or last) edge to a point (at the cursor)
    - left/right/up/down: Move the active vertices
    - ctrl+z: Undo a changes
    - ctrl+shift+z: Redo a previously undone change
    
    We keep a history (in self.history) of all of the past shapes. 
    To add the current shape to the history use self.push() 
    
    """

    def __init__(self, x, y=None, ax=None,
                 complete=False,
                 on_update=None,
                 vertex_select_radius=16):
        """
        Parameters:
        - *x:* A list of points to initialize the polygon with. This may be 
          either a 2D array with each row as a point, or it can be a list 
          with only the x-coordinates (in which case the `y` parameter must 
          also be used)
        - *y:* A list of y-coordinates (only needed if X is a 1D array or list)
        - *ax:* The axis to render onto. 
        - *complete:* Whether we should force the input to be a closed polygon
        - *on_update:* A callback invoked with the `x` and `y` arrays whenever
          they are changed. 
        - *vertex_select_radius:* The distance before the mouse is 'snapped' to a vertex
          (or edge). 
        """
        # Allow the caller to pass in a single 2D array
        if isinstance(x, np.ndarray) and len(x.shape) == 2:
            assert y is None
            assert x.shape[1] == 2
            y = x[:, 1].tolist()
            x = x[:, 0].tolist()

        if isinstance(x, np.ndarray):
            x = x.tolist()

        if isinstance(y, np.ndarray):
            y = y.tolist()

        assert isinstance(x, list)
        assert isinstance(y, list)
        assert len(x) == len(y)

        assert not complete or len(x) >= 3, "complete ==> at least 3 points"

        if len(x) < 1:  # Always need at least one point
            x = x + [0]
            y = y + [0]

        # If the caller wants to force the polygon to be complete
        # then connect the last and first points
        if complete and (x[0] != x[-1] or y[0] != y[-1]):
            x = x + [x[0]]
            y = y + [y[0]]

        self.ax = ax or plt.gca()  # The main axes to render onto
        self.canvas = self.ax.figure.canvas  # The canvas (manages events and rendering)
        self.on_update = on_update

        # The points (we copy-in, in case user changes them)
        self.x = copy(x)
        self.y = copy(y)

        # The list of selected vertices (indexes)
        self._active_vertices = []

        self.vertex_select_radius = vertex_select_radius  # Distance to snap to a vertex

        self.cids = []  # Registered callbacks (saved so they can be disconnected)

        # Keep track of the location of the mouse, and every point,
        # when the mouse is pressed. This is used for dragging points. 
        self._down_event = None
        self._x_at_down = None  # Copy of self.x
        self._y_at_down = None  # Copy of self.y

        # The polygon line
        props = dict(color='k', linestyle='-', linewidth=2, alpha=0.5)
        self.line = self.ax.add_line(Line2D(x, y, **props))

        # The vertex handles
        props = dict(marker='o', markersize=7, mfc='w', markeredgecolor='k',
                     ls='none', alpha=0.5, visible=True, label='_nolegend_')
        self.vertex_markers = self.ax.add_line(Line2D(x, y, **props))

        # The highlighter
        props = dict(marker='o', markersize=7, mfc='y', markeredgecolor='y',
                     ls='-', color='y', lw=2, alpha=1, visible=False, label='_nolegend_')
        self.edge_hl = self.ax.add_line(Line2D([0, 0], [0, 0], **props))

        # Whether the polygon is closed
        self.completed = len(self) >= 3 and self.x[0] == self.x[-1] and self.y[0] == self.y[-1]

        # The state of the widget. Each state handles events differently.
        self.states = ['adding', 'picking', 'dragging']
        if self.completed:
            self.state = 'picking'
        else:
            self.state = 'adding'

        # Keys are mapped to semantic actions [so I can keep them straight]
        self.keymap = {
            'w': 'split-edge',  # Split the active (or last) edge to a point
            'e': 'collapse-edge',  # Collapse the active (or last) edge to a point
            'a': 'select-all',  # Select every vertex & edge
            'left': 'left', 'right': 'right', 'up': 'up', 'down': 'down',  # Move active
            'x': 'constrain-x',
            'y': 'constrain-y',
            'n': 'constrain-normal',  # Slide an edge in its normal direction
            'ctrl+z': 'undo',
            'ctrl+Z': 'redo',
        }

        # Keep track of the editing history
        # - self._stack_position is the index of the last saved state
        # - self.get/set_data accesses the savable state
        self.history = [deepcopy(self.data())]
        self._stack_position = 0

        # Connect events (was refactored into its own method)
        self.connect_events()

        # Draw the initial polygon
        self.update()

    def connect_events(self):
        self.connect('motion_notify_event', self._on_motion)
        self.connect('button_press_event', self._on_button_press)
        self.connect('button_release_event', self._on_button_release)
        self.connect('key_press_event', self._on_key_press)
        self.connect('key_release_event', self._on_key_release)

    def remove(self):
        """Remove all artists and disconnect events"""
        self.disconnect_events()
        self.line.remove()
        self.edge_hl.remove()
        self.vertex_markers.remove()

    def data(self):
        """The savable state of the polygon"""
        return dict(x=self.x, y=self.y, completed=self.completed)

    def set_data(self, state):
        """Restore the polygon to a saved state"""
        self.x[:] = state.get('x', self.x)
        self.y[:] = state.get('y', self.y)
        self.completed = state.get('completed')
        self.update()

    def push(self):
        """Save our state to to the history"""

        # Do not save redundant states
        if (len(self.history) > 0) and self.data() == self.history[-1]:
            return

            # No more redo's available after you make an edit
        if self._stack_position != len(self.history) - 1:
            del self.history[self._stack_position:]
            self._stack_position = len(self.history) - 1

        # Must make sure we are not saving week references 
        self.history.append(deepcopy(self.data()))
        self._stack_position += 1

        # Relic of debugging -- leaving it in for now
        assert (self._stack_position == len(self.history) - 1)

    def undo(self):
        """Restore state to a point in history (moving backward)"""
        if self._stack_position > 0:
            self._stack_position -= 1
            self.set_data(self.history[self._stack_position])
            self.update()

    def redo(self):
        """Restore state to a point in history (moving forward)"""
        if self._stack_position + 1 < len(self.history):
            self._stack_position += 1
            self.set_data(self.history[self._stack_position])
            self.update()

    def reset(self):
        """Restore initial shape"""
        self.set_data(self.history[0])
        self.push()

    def fit_point_in_view(self, x=None, y=None):
        if x is None or y is None:
            x = np.mean([self.x[i] for i in self._active_vertices])
            y = np.mean([self.y[i] for i in self._active_vertices])

        if not self.ax.viewLim.contains(x, y):
            dx = (self.ax.viewLim.xmin + self.ax.viewLim.xmax) / 2. - x
            dy = (self.ax.viewLim.ymin + self.ax.viewLim.ymax) / 2. - y
            self.ax.set_xbound(self.ax.viewLim.xmin - dx, self.ax.viewLim.xmax - dx)
            self.ax.set_ybound(self.ax.viewLim.ymin - dy, self.ax.viewLim.ymax - dy)

    def delete_vertex(self, i: int = None, undoable=True):
        """Delete a vertex (and select the new edge)"""
        if i is None:
            i = self._active_vertices[0]

        del self.x[i]
        del self.y[i]
        self._active_vertices = sorted([(i - 1) % len(self), i])

        if undoable:
            self.push()

    def move_active_vertices(self, dx: float, dy: float, fit_in_view=True, undoable=True):
        """Translate all active vertices by the given vector

        :param dx: X offset (from last committed x)
        :param dy: Y offset (from last committed y)
        :param fit_in_view: Zoom to fit the selected point. This is useful when editing with the keyboard,
                            but not so much when using the mouse.
        :param undoable: Save change to the undo stack. If this is part of a compound edit you should set this False.

        """
        if not self._active_vertices:
            self._active_vertices = list(range(len(self)))

        if self._x_at_down is None:
            x = self.x
            y = self.y
        else:
            x = self._x_at_down
            y = self._y_at_down

        for vi in self._active_vertices:
            self.x[vi] = x[vi] + dx
            self.y[vi] = y[vi] + dy

        if undoable:
            self.push()

        if fit_in_view:
            self.fit_point_in_view()

    def _on_key_press(self, event):
        # Only process events that occur within our axes
        if event.inaxes != self.ax:
            return

        if self.keymap.get(event.key) == 'split-edge':
            self.edge_hl.set_pickradius(self.vertex_select_radius)
            hit, indices = self.edge_hl.contains(event)
            if hit:
                self.split_edge(xy=(event.xdata, event.ydata))
            else:
                self.split_edge()

            self.fit_point_in_view()
            self.push()

        elif self.keymap.get(event.key) == 'collapse-edge':
            if len(self) > 3:
                if len(self._active_vertices) == 1:
                    self.delete_vertex()
                else:
                    hit, indices = self.edge_hl.contains(event)
                    if hit:
                        self.collapse_edge(xy=(event.xdata, event.ydata))
                    else:
                        self.collapse_edge()
                self.fit_point_in_view()
                self.push()

        elif self.keymap.get(event.key) == 'select-all':
            self._active_vertices = list(range(len(self) + 1))

        elif event.key in ('x', 'y', 'n'):
            if self.state in ('dragging', 'adding'):
                self._drag_active(event.xdata, event.ydata, event.key)
                # No need to push history -- will be handled when dragging finishes
        elif event.key == 'left':
            self.move_active_vertices(-1, 0)
        elif event.key == 'right':
            self.move_active_vertices(1, 0)
        elif event.key == 'up':
            self.move_active_vertices(0, -1)
        elif event.key == 'down':
            self.move_active_vertices(0, 1)
        elif event.key == 'ctrl+z':
            self.undo()
        elif event.key == 'ctrl+Z':  # shift + control + z
            self.redo()
        elif event.key == 'pageup':
            self.select_next_edge_or_vertex()
        elif event.key == 'pagedown':
            self.select_next_edge_or_vertex(direction=-1)
        elif event.key == 'shift+pageup':
            self.select_next_edge_or_vertex(single=False)
        elif event.key == 'shift+pagedown':
            self.select_next_edge_or_vertex(single=False, direction=-1)

        self.update()

    def select_next_edge_or_vertex(self, single=True, direction=+1):
        """ Select the next vertex or edge.

        Also makes sure the selection is visible (otherwise I get confused and annoyed)

        :param single:  Only select a single vertex
        :param direction: Move forward (+1) or backward (-1) along the polygon.
        :return: None
        """
        if len(self._active_vertices) == 0:
            self._active_vertices = [0]
        elif len(self._active_vertices) == 1:
            self._active_vertices.append((self._active_vertices[-1] + direction) % len(self))
        elif len(self._active_vertices) == 2:
            v0, v1 = self._active_vertices
            if v0 == (v1 + 1) % len(self):
                v1, v0 = v0, v1
            if direction == +1:
                self._active_vertices = [v1]
            else:
                self._active_vertices = [v0]

        if single:
            self._active_vertices = [self._active_vertices[-1]]
        else:
            self._active_vertices = sorted(self._active_vertices)
        av = self._active_vertices[-1]
        self.fit_point_in_view(self.x[av], self.y[av])

    def _on_key_release(self, event: KeyEvent):
        if event.inaxes != self.ax:
            return

        if event.key in ('x', 'y', 'n'):
            if self.state in ('dragging', 'adding'):
                self._drag_active(event.xdata, event.ydata, '')

    def _on_button_press(self, event: MouseEvent):
        # Only process events that occur within our axes
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # LMB
            if self.completed or event.key == 'control':

                self.state = 'dragging'

                # If nothing is selected, select everything
                if len(self._active_vertices) == 0:
                    self._active_vertices = list(range(len(self) + 1))
            elif self.state == 'adding':
                if len(self) >= 3 and self.x[0] == self.x[-1] and self.y[0] == self.y[-1]:
                    # Close the polygon
                    self.completed = True
                    self.state = 'picking'
                else:
                    # Add a point to the end
                    self.x.append(self.x[-1])
                    self.y.append(self.y[-1])
                self.push()  # Save to history

        # Save information for dragging
        self._down_event = event
        self._x_at_down = copy(self.x)
        self._y_at_down = copy(self.y)

        self.update()

    def _on_button_release(self, event: MouseEvent):
        if not event.inaxes == self.ax:
            return

        if self.state == 'dragging':
            self.push()

        # Clear the stuff we cached when the button was pressed
        self._down_event = None
        self._x_at_down = None
        self._y_at_down = None

        if self.completed:
            self.state = 'picking'
        else:
            self.state = 'adding'
        self.update()

    def _to_window(self, xy):
        return np.array(self.ax.transData.transform(xy))

    def _on_motion(self, event: MouseEvent):
        if event.inaxes != self.ax:
            return

        if self.state == 'picking' or (self.state == 'adding' and event.key == 'control'):
            if self.keymap.get(event.key) == 'select-all':
                # We are selecting every pint
                self._active_vertices = list(range(len(self) + 1))
            else:
                # We are selecting a point or an edge
                self._active_vertices = []
                i, d = self.closest_point((event.x, event.y))
                if i >= 0:
                    self._active_vertices = [i]
                else:
                    i, d = self.closest_edge((event.x, event.y))
                    if i >= 0:
                        self._active_vertices = [i, (i + 1) % len(self)]
        elif self.state == 'dragging' or self.state == 'adding':
            self._drag_active(event.xdata, event.ydata, event.key)

        self.update()

    def _drag_active(self, x, y, key):
        # While we are not complete, we are adding to the end by default
        if not self.completed and key != 'control':
            self._active_vertices = [len(self)]

        if self._down_event is not None:
            lx, ly = self._down_event.xdata, self._down_event.ydata
        else:
            lx, ly = self.get_point(self._active_vertices[0])
        dx = x - lx
        dy = y - ly
        if key == 'n' and len(self._active_vertices) == 2:
            self.slide_edge(self._active_vertices[0], (x, y))
        elif key == 'x':
            self.move_active_vertices(dx, 0, fit_in_view=False, undoable=False)
        elif key == 'y':
            self.move_active_vertices(0, dy, fit_in_view=False, undoable=False)
        else:
            self.move_active_vertices(dx, dy, fit_in_view=False, undoable=False)

        if self.completed:
            self.x[-1] = self.x[0]
            self.y[-1] = self.y[0]
        elif len(self) >= 3:
            # Snap last to first if within radius
            last_point = self._to_window((self.x[-1], self.y[-1]))
            first_point = self._to_window((self.x[0], self.y[0]))
            dist = np.linalg.norm(first_point - last_point)
            if dist < self.vertex_select_radius:
                self.x[-1] = self.x[0]
                self.y[-1] = self.y[0]

    def slide_edge(self, i, xy):
        """ Slide an edge in its normal direction so that it passes through (x,y).

        This preserves all lines except for the line passing through the edge (i,i+1).
        That edge's normal vector is preserved but it is offset to contain the point (x,y).

        :param i: Index of an edge (i, i+1)
        :param xy: target point
        """
        x, y = xy
        i0 = (i - 1) % len(self)
        i1 = i
        i2 = (i + 1) % len(self)
        i3 = (i + 2) % len(self)
        # Four points involved
        p0 = np.array([self._x_at_down[i0], self._y_at_down[i0], 1])
        p1 = np.array([self._x_at_down[i1], self._y_at_down[i1], 1])
        p2 = np.array([self._x_at_down[i2], self._y_at_down[i2], 1])
        p3 = np.array([self._x_at_down[i3], self._y_at_down[i3], 1])
        # Line parallel to edge, passing through mouse
        l1 = np.cross(p1, p2)
        l1[2] = -l1.dot([x, y, 0])
        # Line through incoming edge
        l0 = np.cross(p0, p1)
        # Line through outgoing edge
        l2 = np.cross(p2, p3)
        # New points are at the intersections of the liens
        q1 = np.cross(l0, l1)
        q1 /= q1[2]
        q2 = np.cross(l1, l2)
        q2 /= q2[2]
        # Set the two endpoints
        self.x[i1] = q1[0]
        self.y[i1] = q1[1]
        self.x[i2] = q2[0]
        self.y[i2] = q2[1]

    def highlight(self):
        if len(self._active_vertices) == 0:
            self.edge_hl.set_visible(False)
        else:
            self.edge_hl.set_data(array([(self.x[i], self.y[i]) for i in self._active_vertices]).T)
            self.edge_hl.set_visible(True)

    @staticmethod
    def _distance_to_edge(p0, p1, xy):
        # breakpoint()
        p0 = np.array(p0)
        p1 = np.array(p1)
        xy = np.array(xy)
        v = p1 - p0
        d = np.linalg.norm(v)
        q = v.dot(xy - p0) / d  # Distance along the edge
        q = np.clip(q, 0, d)  # Make sure we are on the line segment
        p = p0 + v * q / d  # Closest point on the line segment
        return np.linalg.norm(xy - p)

    def get_point(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x) - 1

    def __getitem__(self, item):
        return self.get_point(item)

    def closest_edge(self, xy, max_dist=None):
        if max_dist is None:
            max_dist = self.vertex_select_radius

        min_d = max_dist
        min_i = -1

        for i in range(len(self)):
            p0 = self.ax.transData.transform((self.x[i], self.y[i]))
            p1 = self.ax.transData.transform((self.x[i + 1], self.y[i + 1]))
            d = self._distance_to_edge(p0, p1, xy)

            if d <= min_d:
                min_d = d
                min_i = i
        return min_i, min_d

    def closest_point(self, xy, max_dist=16):
        if max_dist is None:
            max_dist = self.vertex_select_radius

        xy = np.array(xy)
        min_i = -1
        min_d = max_dist
        for i in range(len(self)):
            p = self.ax.transData.transform((self.x[i], self.y[i]))
            d = np.linalg.norm(xy - array(p))
            if d <= min_d:
                min_d = d
                min_i = i
        return min_i, min_d

    def split_edge(self, i=None, xy=None):
        # Default to the selected edge or the last edge 
        if i is None:
            if len(self._active_vertices) == 1:
                i = self._active_vertices[0]
            elif len(self._active_vertices) == 2:
                i, j = sorted(self._active_vertices)
                if (j + 1) % len(self) == i:
                    i, j = j, i
                if j != (i + 1) % len(self):
                    print(f"Not an edge:{i},{j}")
                    return
            else:
                return

        # Default to the midpoint
        if xy is None:
            xy = array((self.x[i] + self.x[i + 1], self.y[i] + self.y[i + 1])) / 2.

        # Add a vertex at xy
        self.x.insert(i + 1, xy[0])
        self.y.insert(i + 1, xy[1])

        # Select the newly added vertex
        self._active_vertices = [i + 1]

        self.update()

    def collapse_edge(self, i=None, xy=None):
        # Default to selected edge or the last edge
        if i is None:
            if len(self._active_vertices):
                i = self._active_vertices[0]
            else:
                i = len(self)

        # Default to the midpoint
        if xy is None:
            xy = array((self.x[i] + self.x[i + 1], self.y[i] + self.y[i + 1])) / 2.

        # Collapse to the first point
        del self.x[i + 1]
        del self.y[i + 1]

        i %= len(self)  # Possibly wrap around since we may have been the last edge
        self.x[i] = xy[0]
        self.y[i] = xy[1]

        self._active_vertices = [i]

        # Draw the widget and make sure all markers are synced with the points
        self.update()

    def update(self):
        # Make sure the polygon stays closed
        if self.completed:
            self.x[-1] = self.x[0]
            self.y[-1] = self.y[0]

        # Update the artists

        # Update the outline
        self.line.set_data(self.x, self.y)

        # Update the vertex markers
        self.vertex_markers.set_data(self.x, self.y)

        # Update the highlight
        self.highlight()

        if self.on_update:
            self.on_update(self.x, self.y)

        self.canvas.draw_idle()

    def edge_point(self, i):
        return np.array(((self.x[i + 1] + self.x[i]) / 2., (self.y[i + 1] + self.y[i]) / 2.))

    def edge_vector(self, i):
        return np.array(((self.x[i + 1] - self.x[i]), (self.y[i + 1] - self.y[i])))

    def edge_length(self, i):
        return np.hypot(*self.edge_vector(i))

    def edge_normal(self, i, length=1):
        dx, dy = self.edge_vector(i)
        d = np.hypot(dx, dy)
        return np.array((-length * dy / d, length * dx / d))

    def edge_angle(self, i):
        dx, dy = self.edge_vector(i)
        return np.arctan2(dy, dx)

    def connect(self, event, handler):
        """Connect an event"""
        # Saves the cid so it can be disconnected later
        self.cids.append(self.canvas.mpl_connect(event, handler))

    def disconnect_events(self):
        """Disconnect all event handlers"""
        for cid in self.cids:
            self.canvas.mpl_disconnect(cid)
        self.cids = []
