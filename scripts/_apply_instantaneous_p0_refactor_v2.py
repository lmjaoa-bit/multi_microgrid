from __future__ import annotations

from pathlib import Path
import re

source_path = Path(__file__).with_name("_apply_instantaneous_p0_refactor.py")
source = source_path.read_text(encoding="utf-8")
source = source.replace(
    '    ".github/workflows/matd3-projection-ablation-short-validation.yml",\n',
    '',
)
source = re.sub(
    r'\n# Focused CI should no longer reference the deleted ablation test\.[\s\S]*?(?=\nprint\("instantaneous/P0 refactor applied"\))',
    '\n',
    source,
    count=1,
)
exec(compile(source, str(source_path), "exec"), {"__file__": str(source_path), "__name__": "__main__"})
