"""Verify the supplied local environment data without third-party Python packages."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'data/manifest.json').read_text(encoding='utf-8'))
failed = False
for item in manifest['files']:
    path = root/item['path']
    ok = path.is_file() and path.stat().st_size == item['bytes']
    if ok:
        ok = hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    print(('OK   ' if ok else 'FAIL ')+item['path'])
    failed = failed or not ok
raise SystemExit(1 if failed else 0)
