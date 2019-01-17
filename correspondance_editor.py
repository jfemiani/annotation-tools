from functools import partial

import skimage
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from copy import deepcopy

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import matplotlib.backend_bases
from matplotlib.backend_bases import MouseEvent, KeyEvent
import numpy as np
from numpy.linalg import norm
from skimage.transform import PolynomialTransform

LMB = 0
RMB = 3


def mutator(x):
    def wrapped(self, *args, **kwargs):
        self.begin_editing()
        try:
            x(self, *args, **kwargs)
        finally:
            self.finish_editing()

    return wrapped


class CorrespondenceEditor(object):
    ax: plt.Axes
    canvas: plt.FigureCanvasBase

    def __init__(self, ax, on_commit=None,
                 facecolor=None, edgecolor=None,
                 pick_radius=15):

        self.arrowstyle = "-|>,head_width=2.5, head_length=5"
        self.facecolor = facecolor or (0, 0.5, 1.0, 1.0)
        self.edgecolor = edgecolor or (0, 0.5, 1.0, 1.0)
        self.pick_radius = pick_radius
        self.scroll_speed = -0.5

        # Possible semantics:
        #  An integer > 1 -- fit a polynomial of that order
        #  A float in [0,1] -- Use that fraction of the number of matches as the order
        self.order = 0.5
        self._warp = None
        self._inverse_warp = None

        self.on_finish = on_commit
        self.ax = ax
        self.canvas = ax.figure.canvas

        self.matches = [[], []]

        self._active_point = -1
        self._active_arrow = -1

        self._arrow_patches = []
        self._arrow_tails = [] # Zero-length arrows are iunvisible, so I will put markers at the tails

        self._arrow_highlight = FancyArrowPatch((0, 0), (0, 0), visible=False,
                                                arrowstyle=self.arrowstyle, color='y', linewidth=3)

        self._arrow_highlight_head = Line2D([0], [0], visible=False,
                                            marker='o', markersize=5, markerfacecolor='y', markeredgecolor='y',
                                            pickradius=self.pick_radius)

        self._arrow_highlight_tail = Line2D([0], [0], visible=False,
                                            marker='o', markersize=5, markerfacecolor='y', markeredgecolor='y',
                                            pickradius=self.pick_radius)

        self._point_highlight = Line2D([0], [0], visible=False,
                                       marker='o', markersize=5, markerfacecolor='r', markeredgecolor='r')

        self._arrow_highlight = self.ax.add_patch(self._arrow_highlight)
        self._arrow_highlight_head = self.ax.add_artist(self._arrow_highlight_head)
        self._arrow_highlight_tail = self.ax.add_artist(self._arrow_highlight_tail)

        self._point_highlight = self.ax.add_artist(self._point_highlight)

        self.cids = []

        self._history = []
        self._future = []
        self.save_state()  # Initialuze history (to empty doc)

        self._event = None
        self._mouse_down = None  # Set to the event that started a drag. None if not dragging.
        self._editing = 0  # Whether we are editing. Counts up and down to allow compound edits.

        self.key_handlers = {
            ' ': self.commit,
            's': partial(self.set_active_point, 0),
            't': partial(self.set_active_point, 1),
            'n': partial(self.new_arrow, selected=True, drag_target=False),
            'delete': self.delete_arrow,
            'e': self.delete_arrow,
            'left': partial(self.nudge, dx=-1, dy=0),
            'right': partial(self.nudge, dx=1, dy=0),
            'up': partial(self.nudge, dx=0, dy=-1),
            'down': partial(self.nudge, dx=0, dy=1),
            'ctrl+z': self.undo,
            'ctrl+Z': self.redo,
            'enter': self.commit,
        }

        # Create the arrows
        self.refresh_arrows()

        self.connect_events()

    def __len__(self):
        return len(self.matches[0])

    def __getitem__(self, i):
        return self.matches[0][i], self.matches[1][i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def begin_editing(self):
        """Call before changing the data"""
        self._editing += 1

    def finish_editing(self):
        """Call after you are done modifying the data.
        Saves to history after all edits are complete. """
        assert self._editing > 0
        self._editing -= 1
        if self._editing == 0:
            self._warp = None # We changed the matched --> the warp is dirty
            self._inverse_warp = None # We changed the matched --> the warp is dirty
            self.save_state()
            self.refresh_highlights()
            self.canvas.draw_idle()

    def get_state(self):
        return self.matches

    def set_state(self, state):
        s, t = state
        self.matches[0][:] = s
        self.matches[1][:] = t
        self.refresh_arrows()

    def save_state(self):
        # Sometimes I push multiple times for some reason
        if self._history and self._history[-1] == self.get_state():
            return
        self._history.append(deepcopy(self.get_state()))
        del self._future[:]

    def undo(self):
        if self._history:
            self._future.append(deepcopy(self.get_state()))
            self.set_state(self._history.pop())

    def redo(self):
        if self._future:
            self.set_state(self._future.pop())
            if self._history and self.get_state() != self._history[-1]:
                self._history.append(deepcopy(self.get_state()))

    def connect(self, event, handler):
        self.cids.append(self.canvas.mpl_connect(event, handler))

    def connect_events(self):
        self.connect('motion_notify_event', self._on_motion)
        self.connect('button_press_event', self._on_button_press)
        self.connect('button_release_event', self._on_button_release)
        self.connect('key_press_event', self._on_key_press)
        # self.connect('key_release_event', self._on_key_release)
        self.connect('scroll_event', self._on_scroll)

    def disconnect_events(self):
        """Disconnect all event handlers"""
        for cid in self.cids:
            self.canvas.mpl_disconnect(cid)
        self.cids = []

    def ignore(self, event):
        return event.inaxes != self.ax

    def zoom(self, amount=1, xy=None):
        amount = 2 ** (amount)
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        if xy is None:
            x = (xmin + xmax) / 2.
            y = (ymin + ymax) / 2.
        else:
            x, y = xy
        xmin = x + (xmin - x) * amount
        ymin = y + (ymin - y) * amount
        xmax = x + (xmax - x) * amount
        ymax = y + (ymax - y) * amount
        self.ax.set_xbound(xmin, xmax)
        self.ax.set_ybound(ymin, ymax)

    def _on_scroll(self, event: MouseEvent):
        x, y = event.xdata, event.ydata
        self.zoom(event.step * self.scroll_speed, (x, y))
        self.canvas.draw_idle()

    def _on_motion(self, event: MouseEvent):
        if self.ignore(event):
            return

        if self._mouse_down:
            self.set_point((event.xdata, event.ydata))
        else:
            self.select_arrow(event)

    def _distance_to_line_segment(self, p0, p1, xy, epsilon=1e-4):
        p0 = np.array(p0)
        p1 = np.array(p1)
        xy = np.array(xy)
        v = p1-p0
        v2 = v@v
        if v2 > epsilon:
            q = (v @ (xy-p0))/(v @ v)  # Distance along the edge
            q = np.clip(q, 0, 1) # Make sure we are on the line segment
        else:
            q = 0
        p = p0 + v*q  # Closest point on the line segment
        return np.linalg.norm(xy-p)

    def select_arrow(self, event: MouseEvent):
        patch: FancyArrowPatch
        min_distance = self.pick_radius
        argmin_distance = -1

        p = np.array((event.xdata, event.ydata))
        for i, (s, t) in enumerate(self):
            d = self._distance_to_line_segment(s, t, p)
            if d < min_distance:
                min_distance = d
                argmin_distance = i

        self.set_active_arrow(argmin_distance)

        if 0 <= argmin_distance:
            s, t = self[argmin_distance]
            d0 = norm(p-s)
            d1 = norm(p-t)
            if d0 < d1:
                self.set_active_point(0)
            else:
                self.set_active_point(1)

    def _start_drag(self, event: matplotlib.backend_bases.MouseEvent):
        self._mouse_down = event
        self.begin_editing()
        self.set_point((event.xdata, event.ydata))

    def _on_button_press(self, event: matplotlib.backend_bases.MouseEvent):
        if self.ignore(event):
            return

        self._event = event

        if self.has_active_arrow():
            if event.button == 1:
                # There are a couple of ways we can miss a mouse up event...
                if not self._mouse_down:
                    self._start_drag(event)
        else:
            if event.button == 1:
                self.new_arrow(s=(event.xdata, event.ydata),
                               t=(event.xdata, event.ydata),
                               selected=True,
                               drag_target=True)

    def _on_button_release(self, event: MouseEvent):
        if self.ignore(event):
            return

        if self._mouse_down:
            self._mouse_down = None
            self.finish_editing()

    def _on_key_press(self, event: KeyEvent):
        if self.ignore(event):
            return

        self._event = event
        if event.key in self.key_handlers:
            self.key_handlers[event.key]()

    def refresh_arrows(self):
        # Remove the old arrows
        while self._arrow_patches:
            self._arrow_patches.pop().remove()

        # Dont forget I added a point marker to the back of every arrow
        while self._arrow_tails:
            self._arrow_tails.pop().remove()

        # Add the new ones
        for s, t in zip(*self.matches):
            self._make_arrow_patch(s, t)

        # Refresh the highlights
        self.refresh_highlights()

        # Schedule a redraw
        self.canvas.draw_idle()

    def _set_highlight_arrow(self, s=None, t=None, visible=True):
        if s is not None and t is not None:
            self._arrow_highlight.set_positions(s, t)
            self._arrow_highlight_tail.set_data(s)
            self._arrow_highlight_head.set_data(t)
        self._arrow_highlight.set_visible(visible)
        self._arrow_highlight_head.set_visible(visible)
        self._arrow_highlight_tail.set_visible(visible)

    def refresh_highlights(self):
        if 0 <= self._active_arrow < len(self):
            self._set_highlight_arrow(self.matches[0][self._active_arrow],
                                      self.matches[1][self._active_arrow],
                                      visible=True)
        else:
            self._set_highlight_arrow(visible=False)

        if 0 <= self._active_arrow < len(self) and 0 <= self._active_point < 2:
            self._point_highlight.set_data(*(self.matches[self._active_point][self._active_arrow]))
            self._point_highlight.set_visible(True)
        else:
            self._point_highlight.set_visible(False)

        self.canvas.draw_idle()

    def commit(self, _unused=None):
        if self.on_finish:
            self.on_finish(self)

    def get_active_arrow(self):
        return self._active_arrow

    def set_active_arrow(self, i):
        if i != self._active_arrow:
            self._active_arrow = i
            self.refresh_highlights()

    def has_active_arrow(self):
        return 0 <= self._active_arrow < len(self)

    def get_active_point(self):
        return self._active_point

    def has_active_point(self):
        return 0 <= self._active_point < 2

    def set_active_point(self, i):
        if i != self._active_point:
            self._active_point = i
            self.refresh_highlights()

    def _make_arrow_patch(self, s, t):
        a = FancyArrowPatch(s, t,
                            arrowstyle=self.arrowstyle,
                            facecolor=self.facecolor,
                            edgecolor=self.edgecolor)
        a = self.ax.add_patch(a)
        self._arrow_patches.append(a)

        tail = Line2D([s[0]], [s[1]],
                      marker='o',
                      markersize=2.5,
                      markeredgecolor=self.edgecolor,
                      markerfacecolor=self.facecolor,
                      pickradius=self.pick_radius
                      )
        tail = self.ax.add_artist(tail)
        self._arrow_tails.append(tail)


    @mutator
    def new_arrow(self, s=None, t=None, selected=True, drag_target=False):
        # Default to the location of the mouse in most recent event
        if s is None:
            if t is None:
                s = self._event.xdata, self._event.ydata
            else:
                s = self.predict_source(t)
        if t is None:
            if s is None:
                t = self._event.xdata, self._event.ydata
            else:
                t = self.predict_target(s)

        # Add the new source and target
        self.matches[0].append(s)
        self.matches[1].append(t)

        # Add a patch for the new arrow
        self._make_arrow_patch(s, t)

        # Users expect the new arrow to be selected
        if selected:
            self.set_active_arrow(len(self) - 1)
            self.set_active_point(1)  # Presumably we clicked on the tail and we will click on the head next

        if drag_target:
            self._start_drag(self._event)

    # noinspection PyUnusedLocal
    @mutator
    def set_point(self, xy, point=None, arrow=None):
        patch: FancyArrowPatch
        line: Line2D

        if arrow is None:
            arrow = self._active_arrow

        if point is None:
            point = self._active_point

        if not 0 <= arrow < len(self):
            return  # No arrows to select yet

        self.matches[point][arrow] = xy

        # Update the plot elements for the arrow
        patch = self._arrow_patches[arrow]
        patch.set_positions(self.matches[0][arrow], self.matches[1][arrow])

        # And also move the tail
        self._arrow_tails[arrow].set_data(self.matches[0][arrow])

        # Update the highlights if we are moving the selected item
        if arrow == self._active_arrow:
            self.refresh_highlights()

    def ensure_selected_arrow(self):
        if len(self) == 0:
            self.new_arrow()

        if self._active_arrow < 0:
            self.set_active_arrow(len(self) - 1)

    def ensure_selected_point(self):
        self.ensure_selected_arrow()
        if self._active_point < 0:
            self.set_active_arrow(len(self) - 1)

    @mutator
    def nudge(self, dx=0, dy=0):
        self.ensure_selected_point()
        x, y = self.matches[self._active_point][self._active_arrow]
        self.set_point((x + dx, y + dy))

    @mutator
    def delete_arrow(self, arrow=None):
        if arrow is None:
            self.ensure_selected_arrow()
            arrow = self._active_arrow

        del self.matches[0][arrow]
        del self.matches[1][arrow]

        # Remove the patch from the plot
        arrow_patch = self._arrow_patches.pop(arrow)
        arrow_patch.remove()

        tail = self._arrow_tails.pop(arrow)
        tail.remove()

        # Update the active arrow index (it might have shifted)
        # The behavior if we delete the active arrow should be that
        # the next arrow is selected. Otherwise the selected arrow
        # should be the same.
        if self._active_arrow > arrow:
            self._active_arrow -= 1

    def get_sources(self):
        return self.matches[0]

    def get_targets(self):
        return self.matches[1]

    def get_transform_order(self):
        if self.order > 1:
            order = min(self.order, len(self))
        else:
            order = round(self.order * len(self))
        return order

    def get_warp(self, recompute=False):
        if recompute:
            self._warp = None

        if self._warp is None:
            self._warp = PolynomialTransform()
            self._warp.estimate(np.array(self.get_sources()),
                                np.array(self.get_targets()),
                                self.get_transform_order())

        return self._warp

    def get_inverse_warp(self, recompute=False):
        if recompute:
            self._inverse_warp = None

        if self._inverse_warp is None:
            self._inverse_warp = PolynomialTransform()
            self._inverse_warp.estimate(self.get_targets(),
                                        self.get_sources(),
                                        self.get_transform_order())

        return self._warp

    def predict_target(self, s):
        t = self.get_warp()(np.array([s]))[0]
        return tuple(t)

    def predict_source(self, t):
        s =  self.get_inverse_warp()(np.array([t]))[0]
        return tuple(s)



# noinspection PyUnusedLocal
def demo():
    fig: plt.Figure
    ax: plt.Axes

    fig = plt.figure()

    # Prevent the toobar from handling some key press events (e.g. s)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

    ax = plt.axes()

    im_left, im_right, disparity = skimage.data.stereo_motorcycle()
    h, w, _ = im_left.shape
    ax.imshow(im_left, extent=[-w - 0.5, -0.5, h - 0.5, -0.5])
    ax.imshow(im_right, extent=[-0.5, w - 0.5, h - 0.5, -0.5])

    ax.set_xbound(-w - 0.5, w + 0.5)

    def handle_commit(ce):
        print("Committed", ce.matches)

    ce = CorrespondenceEditor(ax, on_commit=handle_commit)

    return ce


if __name__ == '__main__':
    ce = demo()
    plt.show()
