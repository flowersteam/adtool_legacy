from inspect import getmembers, isclass
from importlib import import_module
import pkgutil

def list_classes(module_name, prefix=""):
    imported_module = import_module(module_name)
    classes = []
    for member in getmembers(imported_module, isclass):
        classes.append({
            'name': prefix + member[0],
            'class': member[1]
        })
    for sub_module in pkgutil.iter_modules(imported_module.__path__):
        if sub_module.ispkg:
            classes.extend(
                list_classes(module_name + "." + sub_module.name, 
                             prefix=sub_module.name + '.')
            )

    return classes


# import sys
# import inspect
# def list_classes(module_name):
#     classes = []
#     mymodule = import_module(module_name)
#     for element_name in dir(mymodule):
#         element = getattr(mymodule, element_name)
#         if inspect.isclass(element):
#             classes.append({
#             'name': element_name,
#             'class': element
#             })
    
#     return classes