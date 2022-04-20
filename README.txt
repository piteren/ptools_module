ptools Module project (for python 3.7)


to install ptools as a module in your python project, use command like:
pip install git+https://github.com/piteren/ptools_module.git#egg=ptools

or add to requirements.txt:
git+https://github.com/piteren/ptools_module.git#egg=ptools


below some more options:

# explicitly state the package name
git+https://github.com/piteren/ptools_module.git#egg=ptools

# specify commit hash
git+https://github.com/piteren/ptools_module.git@d48018d#egg=ptools

# specify branch name
git+https://github.com/piteren/ptools_module.git@master#egg=ptools

# specify tag
git+https://github.com/piteren/ptools_module.git@v0.9#egg=ptools


to create annotated tag:
git tag -a vX.X.X -m "annotation text"