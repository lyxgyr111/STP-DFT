"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32 --head_dims="[64,32,16,16]"
This version is modified to be more flexible for research experiments:
- It can INTRODUCE new config keys from the command line (e.g., head_dims).
- It removes the strict type-matching assertion, allowing overrides of
  defaults like `None` with specific types like `list`.
"""
import sys
from ast import literal_eval
for arg in sys.argv[1:]:
    if '=' not in arg:
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
for arg in sys.argv[1:]:
    if '=' in arg:
        assert arg.startswith('--')
        key, val = arg.split('=', 1) 
        key = key[2:]
        try:
            attempt = literal_eval(val)
        except (SyntaxError, ValueError):
            attempt = val
        print(f"Overriding/Setting: {key} = {attempt} (type: {type(attempt)})")
        globals()[key] = attempt