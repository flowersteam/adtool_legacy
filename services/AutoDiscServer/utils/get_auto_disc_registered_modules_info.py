import math

def check_jsonify(my_dict):
    try:
        for key, value in my_dict.items():
            if isinstance(my_dict[key], dict):
                check_jsonify(my_dict[key])
            elif isinstance(my_dict[key], list):
                for x in my_dict[key]:
                    if isinstance(my_dict[key], dict):
                        check_jsonify(x)
                    elif isinstance(my_dict[key], float):
                         if math.isinf(my_dict[key]):
                            if my_dict[key] > 0:
                                my_dict[key] = "inf"
                            else:
                                my_dict[key] = "-inf"
            elif isinstance(my_dict[key], float):
                if math.isinf(my_dict[key]):
                    if my_dict[key] > 0:
                        my_dict[key] = "inf"
                    else:
                        my_dict[key] = "-inf"
    except Exception as ex:
        print("my_dict = ", my_dict, "key = ", key, "my_dict[key] = ",my_dict[key] , "exception =", ex)

def get_auto_disc_registered_modules_info(registered_modules):
    infos = []
    for module_name, module_class in registered_modules.items():
        info = {}
        info["name"] = module_name
        info["config"] = module_class.CONFIG_DEFINITION
        if hasattr(module_class, 'input_space'):
            info['input_space'] = module_class.input_space.to_json()

        if hasattr(module_class, 'output_space'):
            info['output_space'] = module_class.output_space.to_json()

        if hasattr(module_class, 'step_output_space'):
            info['step_output_space'] = module_class.step_output_space.to_json()
        check_jsonify(info)
        infos.append(info)
    return infos

def get_auto_disc_registered_callbacks(registered_callbacks):
    infos = []
    for category, callbacks_list in registered_callbacks.items():
        info = {}
        info["name_callbacks_category"] = category
        info["name_callbacks"] = list(callbacks_list.keys())
        infos.append(info)
    return infos