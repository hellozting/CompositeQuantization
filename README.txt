###################################################################
#                                                                 #
#  Composite Quantization for Approximate Nearest Neighbor Search #
#                                                                 #   
#              Ting Zhang (zting@mail.ustc.edu.cn)                #
#                Chao Du (duchao0726@gmail.com)                   #
#             Jingdong Wang (jingdw@microsoft.com)                #
#                                                                 #
###################################################################
-----------
What is it?
-----------

This software library implements the composite quantization algorithm
described in

      Composite Quantization for approximate nearest neighbor search. 
      In International Conference on Machine Learning (ICML), 2014.

If you use this software for research purposes, you should cite the aforementioned paper in any resulting publication.

------------------
The latest version
------------------
Version 1.0 (2015-01-28):
	Initial release.

------------
Installation
------------
See HOW_TO_INSTALL.txt.

-------------
Documentation
-------------
See HOW_TO_TRAINING.txt and HOW_TO_SEARCH.txt.

-------------
Example usage
-------------
See demo.cpp and config.txt in the source code.

---------
Licensing
---------
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


---------
Reference
---------
libLBFGS: 
          http://www.chokkan.org/software/liblbfgs/index.html
          
Fast k-means: 
          http://research.microsoft.com/en-us/um/people/jingdw/LargeScaleClustering/index.html

Product quantization:
          http://people.rennes.inria.fr/Herve.Jegou/projects/ann.html
