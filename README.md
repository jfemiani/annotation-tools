# Tools for editing labeled GSV images

## Overview 

It is easier to demo than to write documentation at this phase of development -- although I should make a keymap. A demo video will be put on Google drive and I will link to it after it uploads. 

Here is the video:
[Quick and poorly done demo of the software](https://drive.google.com/a/miamioh.edu/file/d/1GSLZ6SxRTNUJhUpCXeLIq11hDzIqoqwP/view?usp=sharing)

## Data
The data is on google drive -- you will need to ask for permission from me  (for now) until the data is considered public.
Once you obtain access you can put it in the correct (default) location by this:
```bash
 cd $REPO_FOLDER
 unzip merged.zip
```
where `$REPO_FOLDER` should be replaced by (not surprisingly) the folder that you have cloned this repo into. 

## Running the program

### From a terminal:
```bash
python annotationeditor.py --help
```
and read the instructions. 

### From within a Jupyter Notebook

```python3
%pylab widget 
# or %matplotlib widget iof you prefer

from annotationeditor import AnnotationEditor
fig = figure()
ax = gca()
ae = AnnotationEditor(ax)

# Create a labelbox to add the labels to the bottom
ae.create_label_box()
```
Note that you need to keep the `AnnotationEditor` object alive for the figuire to remain responsive. 

## Keymap

- Hovering over an item highlights it
- Setting the active label (e.g. through the label box) toggles object filtering; when filtering is on, only objects with that label are shown. When filtering is off, everything is selectable. 
- <kbd>LMB</kbd> to select an item
- <kbd>RMB</kbd> to start dragging / panning the view
- <kbd>CTRL</kbd> <kbd>wheel</kbd> to zoom in and out around the mouse cursor
- <kbd>p</kbd> starts pan mode 
- <kbd>+</kbd> zooms in to mouse cursor, and <kbd>-</kbd> zooms out
- <kbd>4</kbd>,<kbd>8</kbd>,<kbd>6</kbd>,<kbd>2</kbd> pan left, up, right, down. 
- <kbd>PgUp</kbd> and <kbd>PgDn</kbd> zoom to the next or previous label. 
- <kbd>SPACE</kbd> starts (or finishes) editing a polygon (key bindings are different when editing).  
- <kbd>ESC</kbd> Clear selection
- <kbd>*</kbd> Toggle visibility of all objects 
- <kbd>5</kbd> Zoom to selected object
- <kbd>CTRL+Z</kbd> Undo a change (undoes an entire edit to an object, can also undo object insertion, deletion, can undo moving to the next/previous file also)
- <kbd>CTRL+SHIFT+Z</kbd> Redo a previously undone change
- <kbd>s</kbd> Save the annotation file
- <kbd>d</kbd> Duplicate the selected object. This works in edit mode or regular mode. 
- <kbd>e</kbd> Erase an object, if not currentlky editing, 
- <kbd>SHIFT + PgUp</kbd> Load the next annotation file (saves the current one first), unless we are currently in editing mode. 
- <kbd>SHIFT + PgDn</kbd> Load the previous annotation file (saves the current one first), unless we are currently in editing mode. 
- <kbd>BACKSLASH </kbd> Rename the selected object to mark it 'unlabeled'

A different keymap applies when the object editor is active
- Setting the active label (e.g. using the label box) changes the label of the edited object. 
- Hovering with the mouse selects points or edges
- <kbd>LMB</kbd> drags the selected point or edge, if nothing is selects you will drag the entire polygon. 
- <kbd>n</kbd> + <kbd>LMB</kbd> Slide selected edge in perspective (the line moves in its normal direction, and stretches to fit the lines indicent to it)
- <kbd>x</kbd> + <kbd>LMB</kbd> or <kbd>x</kbd> + <kbd>LMB</kbd> constrain motion to the horizontal (x) or virtical (y) directrions. 
- <kbd>LEFT</kbd>, <kbd>RIGHT</kbd>, <kbd>UP</kbd>, <kbd>DOWN</kbd> move the selection one pixel at a time. 
- <kbd>PgUp</kbd> Select next point
- <kbd>PgDn</kbd> Select previous point
- <kbd>SHIFT</kbd> + <kbd>PgUp</kbd> Select next edge or point
- <kbd>SHIFT</kbd> + <kbd>PgDn</kbd> Select previous edge or point
- <kbd>CTRL+z</kbd>  Undo an edit to the object.
- <kbd>CTRL+SHIFT+z</kbd>  Redo an edit to the object.



## Palette
The color pallette is in a file (gsv24.palette). It was generated using the [glasbey](https://github.com/taketwo/glasbey) method of gnerating maximally distinct colors for categorical images. 

Some palette items were rearranged / edited for aesthetics. 

## Class/Label Names
The class names are in a text (gsv24.names).  They are listed in our best estimate for their ZORDER -- that is, rendering them from first-to-last using the painters algorithm is a reasonable way of producing a single label per pixel. 

 

