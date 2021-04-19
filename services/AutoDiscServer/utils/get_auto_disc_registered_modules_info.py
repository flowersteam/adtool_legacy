def get_auto_disc_registered_modules_info(registered_modules):
    info = {}
    for module_name, module_class in registered_modules.items():
        info[module_name] = {}
        info[module_name]['config'] = module_class.CONFIG_DEFINITION
        if hasattr(module_class, 'input_space'):
            info[module_name]['input_space'] = module_class.input_space.to_json()

        if hasattr(module_class, 'output_space'):
            info[module_name]['output_space'] = module_class.output_space.to_json()

        if hasattr(module_class, 'step_output_space'):
            info[module_name]['step_output_space'] = module_class.step_output_space.to_json()
    return info
