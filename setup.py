from setuptools import setup, find_packages

from helpers import get_requirements


setup(
    name=               'ptools',
    url=                'https://github.com/piteren/ptools_module.git',
    author=             'Piotr Niewinski',
    author_email=       'pioniewinski@gmail.com',
    packages=           find_packages(),
    version=            'v0.9.2',
    install_requires=   get_requirements(),
    license=            'MIT',
    license_files =     ('license.txt',),
    description=        'python tools (ptools) by piteren',
    long_description=   open('README.txt').read())