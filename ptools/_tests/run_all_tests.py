import os
import unittest

script_dir = os.path.dirname(os.path.abspath(__file__))
dirs = [d for d in os.listdir(script_dir) if os.path.isdir(f'{script_dir}/{d}')]
start_dirs = [d for d in dirs if not d.startswith('_')]
#print(start_dirs)
# TODO: add recurrence for dirs discovery

for start_dir in start_dirs:
    loader = unittest.TestLoader()
    suite = loader.discover(f'{script_dir}/{start_dir}')
    runner = unittest.TextTestRunner()
    runner.run(suite)