# HooBAS


Hoomd Blue initial configurator for Arbitrary Shapes 

Copyright (C) 2016

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public 
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.



Generates an initial system of block-DNA objects for hoomd. The block can be any shape (sphere, cube, ..., custom) 
and is able to load from pdb files. Current implementation is based on a few objects : CenterFile (generates center positions), 
Genshape (generate shape and directives) and Build (assembles the system from center and shapes). 

Since it is based on much older code that wasn't meant to be used in this way and has been maintaining legacy compatibilty with
that code, it is currently implemented in a bizarre way, with many obsolete function, options and names. Newer versions may break
compatibility with legacy versions when this is corrected. 
