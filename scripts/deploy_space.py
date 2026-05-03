"""Push the hf_space/ bundle to a HuggingFace Space.

Pure-Python — no bash, no git operations. Works in any terminal.

Auth: set HF_TOKEN env var, or run `huggingface-cli login` first.

Usage:
    python scripts/deploy_space.py FerasMad
    python scripts/deploy_space.py FerasMad my-custom-space-name
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi, whoami

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "hf_space"


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/deploy_space.py <hf-username> [space-name]")
        return 1

    user = sys.argv[1]
    space_name = sys.argv[2] if len(sys.argv) > 2 else "arabic-complaints-classifier"
    space_id = f"{user}/{space_name}"

    if not SOURCE.exists():
        print(f"ERROR: hf_space/ not found at {SOURCE}")
        return 1

    print("Checking authentication...")
    try:
        info = whoami()
        print(f"  Logged in as: {info['name']}")
    except Exception as e:
        print(f"ERROR: not authenticated. Set HF_TOKEN or run `huggingface-cli login`.")
        print(f"  Detail: {e}")
        return 1

    api = HfApi()

    files = sorted(p.name for p in SOURCE.iterdir() if p.is_file())
    total_kb = sum(p.stat().st_size for p in SOURCE.iterdir() if p.is_file()) / 1024
    print(f"\nUploading {len(files)} files ({total_kb:.0f} KB) to Space {space_id}")
    for f in files:
        print(f"  {f}")

    api.upload_folder(
        folder_path=str(SOURCE),
        repo_id=space_id,
        repo_type="space",
        commit_message="Initial deploy",
    )
    print(f"\nDone. Space: https://huggingface.co/spaces/{space_id}")
    print("Build takes ~3-5 min. First inference cold-starts in ~30s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
