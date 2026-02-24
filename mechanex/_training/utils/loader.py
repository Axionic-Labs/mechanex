import importlib.util
import os
import sys

def load_user_class(file_path: str, class_name: str):
    """Dynamically load a class from a given file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"User file not found: {file_path}")
    
    module_name = os.path.basename(file_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    if not hasattr(module, class_name):
        raise AttributeError(f"Class {class_name} not found in {file_path}")
    
    return getattr(module, class_name)
