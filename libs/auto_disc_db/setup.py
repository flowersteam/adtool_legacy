from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor in [6, 7] , \
    "This repo is designed to work with Python 3.6 or 3.7." \
    + "Please install it before proceeding."

setup(
    name='auto_disc_db',
    py_modules=['auto_disc_db'],
    version="1.0",
    install_requires=[
        'requests==2.28.1',
        'json'
    ],
    description="auto_disc_db Python lib",
    author="Flowers Team Inria",
)