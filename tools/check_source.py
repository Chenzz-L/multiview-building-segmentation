"""Compile source without importing training dependencies or writing bytecode."""
import ast
from pathlib import Path
root = Path(__file__).resolve().parents[1]
files = sorted(p for p in root.rglob('*.py') if not any(x in p.parts for x in ('.venv', 'runtime', '.git')))
for path in files:
    compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')
print(f'PASS: {len(files)} Python source files parsed')
