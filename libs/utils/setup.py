from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor in [6, 7], \
    "This repo is designed to work with Python 3.6 or 3.7." \
    + "Please install it before proceeding."

setup(
    name='flowers-utils',
    py_modules=['leaf', 'depinj', 'leafutils', 'filetype_converter'],
    version="1.0",
    install_requires=["sqlalchemy==2.0.3"],
    description="utils for flowers",
    author="Flowers Team Inria",
)
