"""Skill 注入：把领域专家经验作为 system prompt 的附加段落。

这是"便宜模型 + 领域经验 ≈ 贵模型"假设的最小实现（DESIGN.md §Experiment）：
skill 文件是纯 markdown，按 mini 化的 Claude Code skills 思路组织——
每条经验都应能追溯到一类真实失败（棘轮原则）。
"""
from __future__ import annotations

from pathlib import Path

from .config import PROJECT_ROOT

SKILLS_DIR = PROJECT_ROOT / "skills"


def load_skill(name_or_path: str) -> str:
    """按名字（skills/<name>.md）或路径加载 skill 内容。"""
    p = Path(name_or_path)
    if not p.exists():
        p = SKILLS_DIR / f"{name_or_path.removesuffix('.md')}.md"
    if not p.exists():
        available = sorted(f.stem for f in SKILLS_DIR.glob("*.md")) if SKILLS_DIR.exists() else []
        raise FileNotFoundError(
            f"skill '{name_or_path}' not found; available: {available or '(none)'}")
    return p.read_text(encoding="utf-8")


def render_skills_section(names: list[str]) -> str:
    if not names:
        return ""
    parts = ["\n\n# Domain knowledge\n"
             "Expert guidance for this kind of task. Follow it unless it conflicts "
             "with the user's explicit instructions."]
    for name in names:
        parts.append(load_skill(name).strip())
    return "\n\n".join(parts)
