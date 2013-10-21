Viscosaur Developer Reference
=============================

Tools
-----
Some important tools to know about for the development of viscosaur:

**Git** is used for distributed version control. 
The git repository is located at https://github.com/tbenthompson/viscosaur.
A huge list of useful information for git beginners is on 
`stackoverflow <http://stackoverflow.com/questions/315911/git-for-beginners-the-definitive-practical-guide/1350157#1350157/>`_

**Sphinx** is used for this documentation and the documentation is the "docs" folder. 
If you need to generate the documentation,
Sphinx can be installed in Ubuntu using the command::
    
    sudo apt-get install python-sphinx

**Bugs Everywhere** is used for issue and bug tracking and for storing ideas for the future.
Bugs Everywhere can be downloaded from http://bugseverywhere.org/. A detailed
description of the various installation methods is found at this 
`link <http://docs.bugseverywhere.org/master/install.html/>`_. 
To install, I downloaded the latest stable source, modified the Makefile
to install to "/usr/local" and then ran::

    make
    make install
