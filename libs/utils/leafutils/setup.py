from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor in [6, 7], \
    "This repo is designed to work with Python 3.6 or 3.7." \
    + "Please install it before proceeding."

setup(
    name="leafutils",
    py_modules=["leafstructs", "leafintegrations"],
    version="1.0",
    description="structs for leaf architecture",
    author="Flowers Team Inria",
)
