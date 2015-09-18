# HooBAS
Hoomd Blue initial configurator for Arbitrary Shapes 

Generates an initial system of block-DNA objects for hoomd. The block can be any shape (sphere, cube, ..., custom) 
and is able to load from pdb files. Current implementation is based on a few objects : CenterFile (generates center positions), 
Genshape (generate shape and directives) and Build (assembles the system from center and shapes). 

Since it is based on much older code that wasn't meant to be used in this way and has been maintaining legacy compatibilty with
that code, it is currently implemented in a bizarre way, with many obsolete function, options and names. Newer versions may break
compatibility with legacy versions when this is corrected. 
