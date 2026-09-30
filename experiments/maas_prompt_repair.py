"""Restore literal LaTeX and escape non-placeholder braces in native CoT template."""
import ast
import re

def repaired_cot(source):
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="GENERATE_COT_PROMPT" for t in n.targets))
    literal=ast.get_source_segment(source,node.value)
    # Upstream uses a plain triple-quoted literal; preserve its written LaTeX bytes.
    assert literal.startswith("'''") and literal.endswith("'''")
    text=literal[3:-3]
    text=text.replace("{","{{").replace("}","}}")
    for field in ("instruction","input"):
        text=text.replace("{{"+field+"}}","{"+field+"}")
    return text

def install(module, source):
    module.GENERATE_COT_PROMPT=repaired_cot(source)


def install_instructions(module, source):
    """Restore written LaTeX in native instruction strings (not format templates)."""
    changed = []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        literal = ast.get_source_segment(source, node.value)
        if not (literal.startswith('"""') and literal.endswith('"""')):
            raise ValueError("Unexpected native instruction string syntax")
        restored = literal[3:-3]
        for target in node.targets:
            if not isinstance(target, ast.Name):
                raise ValueError("Unexpected native instruction assignment")
            if getattr(module, target.id) != restored:
                setattr(module, target.id, restored)
                changed.append(target.id)
    return changed
