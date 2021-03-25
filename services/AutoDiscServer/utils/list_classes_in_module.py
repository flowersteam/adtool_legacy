from inspect import getmembers, isclass
from importlib import import_module

def list_classes(module_name):
    imported_module = import_module(module_name)
    classes = []
    for member in getmembers(imported_module, isclass):
        classes.append({
            'name': member[0],
            'class': member[1]
        })
    return classes