from distutils.core import setup

setup(
    name=               'ptools',
    url=                'https://github.com/piteren/ptools_module.git',
    author=             'Piotr Niewinski',
    author_email=       'pioniewinski@gmail.com',
    packages=           [
        'ptools',
        'ptools.pms',
        'ptools.pms.hpmser'],
    #version=            'v0.9',
    install_requires=   [
        'GPUtil==1.4.0',
        'Levenshtein==0.16.0',
        'matplotlib==3.3.3',
        'nltk==3.5',
        'numpy==1.19.5',
        'pandas==1.1.5',
        'plotly==4.14.1',
        'psutil==5.9.0',
        'regex==2020.11.13',
        'rouge==1.0.0',
        'scipy==1.5.4',
        'sklearn==0.0',
        'spacy==2.3.4',
        'tensorflow==2.6.2',
        'typing-extensions==3.7.4.3'],
    license=            'Creative Commons Attribution-Noncommercial-Share Alike license',
    description=        'python tools (ptools) by piteren',
    long_description=   open('README.txt').read())