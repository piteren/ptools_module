from setuptools import setup, find_packages

from helpers import get_requirements


setup(
    name=               'ptools',
    url=                'https://github.com/piteren/ptools_module.git',
    author=             'Piotr Niewinski',
    author_email=       'pioniewinski@gmail.com',
    packages=           find_packages(),
    version=            'v0.9.4',
    install_requires=   get_requirements(),
    python_requires=    '>=3.7',
    license=            'MIT',
    license_files =     ('LICENSE.txt',),
    description=        'python tools (ptools) by piteren')