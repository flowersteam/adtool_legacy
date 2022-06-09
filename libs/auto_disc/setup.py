from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor in [6, 7] , \
    "This repo is designed to work with Python 3.6 or 3.7." \
    + "Please install it before proceeding."

setup(
    name='auto_disc',
    py_modules=['auto_disc'],
    version="1.0",
    install_requires=[
        'addict',
        'matplotlib',
        'numpy',
        'pillow',
        'requests',
        'graphviz',
        'neat-python==0.92',
        'torch==1.7.1',
        'tinydb',
        'imageio==2.9.0',
        'imageio-ffmpeg'
    ],
    description="auto_disc Python lib",
    author="Flowers Team Inria",
)