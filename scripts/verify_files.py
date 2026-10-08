"""Verify relative-file SHA-256 manifests for source or model assets."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    errors = []
    for name, expected in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(name + ': missing or invalid path')
            continue
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            errors.append(name + ': hash mismatch')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f"Verified {len(manifest['files'])} files ({manifest['scope']}).")


if __name__ == '__main__':
    main()
