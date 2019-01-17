# Tools for editing labeled GSV images

## Overview 

It is easier to demo than to write documentation at this phase of development -- although I should make a keymap. A demo video will be put on Google drive and I will link to it after it uploads. 

## Data
The data is on google drive -- you will need to ask for permission from me  (for now) until the data is considered public. 

## Palette
The color pallette is in a file (gsv24.palette). It was generated using the [glasbey](https://github.com/taketwo/glasbey) method of gnerating maximally distinct colors for categorical images. 

Some palette items were rearranged / edited for aesthetics. 

## Class/Label Names
The class names are in a text (gsv24.names).  They are listed in our best estimate for their ZORDER -- that is, rendering them from first-to-last using the painters algorithm is a reasonable way of producing a single label per pixel. 


## Keymap

- Hovering over an item highlights it
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
- <kbd>CTRL+Z</kbd> Undo a change
- <kbd>CTRL+SHIFT+Z</kbd> Redo a previously undone change
- <kbd>s</kbd> Save the annotation file
- <kbd>d</kbd> Duplicatre the selected object
- <kbd>e</kbd> Erase an object
- <kbd>SHIFT + PgUp</kbd> Load the next annotation file (saves the current one first)
- <kbd>SHIFT + PgDn</kbd> Load the previous annotation file (saves the current one first)
- <kbd>BACKSLASH </kbd> Rename the selected object to mark it 'unlabeled'

A different keymap applies when the object editor is active
(TODO)






 

