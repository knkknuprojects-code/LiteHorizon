"""Conservative local checks before a GitHub upload (stdlib only).

Not a substitute for secret scanning, data licensing, or a human audit.
Run: python scripts/check_public_release.py
"""
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
expected = [
    '.gitignore', '.gitattributes', 'README.md', 'LICENSE', 'CITATION.cff',
    'CONTRIBUTING.md', 'SECURITY.md',
    'docs/INSTALL.md', 'docs/QUICKSTART.md', 'docs/REPRODUCIBILITY.md',
    'paper/EVIDENCE_MATRIX.md', '.github/workflows/ci.yml',
]
missing = [name for name in expected if not (root / name).is_file()]
if missing:
    print('Missing publication files:', *missing, sep='\n - ')
    sys.exit(1)

if (root / '.git').is_dir():
    result = subprocess.run(['git', 'ls-files'], cwd=root, capture_output=True, text=True)
    tracked = result.stdout.splitlines() if result.returncode == 0 else []
    violations = []
    for f in tracked:
        parts = Path(f).parts
        if f in ('.horizon.env', '.env') or parts[0] in (
            'data','corpus','raw','records','runs','models','adapters','bench'
        ):
            violations.append(f)
        if parts[0] in ('collection','datasets','patents','reports') and f != parts[0] + '/README.md':
            violations.append(f)
        if Path(f).suffix.lower() in ('.xlsx','.xls','.safetensors','.ckpt','.p12','.pfx'):
            violations.append(f)
    if violations:
        print('Potentially unsafe tracked files:', *sorted(set(violations)), sep='\n - ')
        sys.exit(2)

print('PASS: expected release files exist and no obviously forbidden tracked paths were found.')
print('NOTE: This check does not establish code authorship, redistributability or absence of secrets.')
