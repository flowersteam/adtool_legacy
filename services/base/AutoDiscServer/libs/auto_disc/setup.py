from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor in [6, 7], \
    "This repo is designed to work with Python 3.6 or 3.7." \
    + "Please install it before proceeding."

setup(
    name='auto_disc',
    py_modules=['auto_disc', 'auto_disc_legacy'],
    version="1.0",
    install_requires=[
        'addict==2.4.0',
        'matplotlib==3.5.3',
        'numpy==1.21.6',
        'pillow==9.2.0',
        'requests==2.28.1',
        'graphviz==0.20.1',
        'neat-python==0.92',
        'torch==1.7.1',
        'tinydb==4.7.0',
        'imageio==2.9.0',
        'imageio-ffmpeg==0.4.7'
    ],
    description="auto_disc Python lib",
    author="Flowers Team Inria",
)
