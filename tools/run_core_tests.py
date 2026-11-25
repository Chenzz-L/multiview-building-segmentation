"""Run existing parameter-free fusion tests without requiring pytest."""
import importlib.util
import inspect
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import torch
torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('core_tests',root/'tests/test_sail_image_memory.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
tests=[(n,f) for n,f in inspect.getmembers(module,inspect.isfunction) if n.startswith('test_')]
for name,fn in tests:
    assert not inspect.signature(fn).parameters, 'fixture-dependent test needs pytest: '+name
    fn();print('PASS',name)
print(f'PASS: {len(tests)} fusion tests')
