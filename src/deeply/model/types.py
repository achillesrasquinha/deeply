import inspect

def is_layer_type(type_, name):
    result = False

    if inspect.isclass(type_):
        type_name = type_.__name__

        if type_name.startswith("Conv") and name == "convolution":
            result = True

    return result