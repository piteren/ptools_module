from setuptools import setup, find_packages

# reads requirements from 'ptools/requirements.txt'
def get_requirements():
    with open('ptools/requirements.txt') as file:
        lines = [l[:-1] for l in file.readlines()]
        return lines


setup(
    name=               'ptools',
    url=                'https://github.com/piteren/ptools_module.git',
    author=             'Piotr Niewinski',
    author_email=       'pioniewinski@gmail.com',
    packages=           find_packages(),
    version=            'v0.9.4',
    install_requires=   get_requirements(),
    license=            'MIT',
    license_files =     ('LICENSE.txt',),
    description=        'python tools (ptools) by piteren')