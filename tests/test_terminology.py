"""Keep domain vocabulary consistent; exceptions belong to explicit conversion input.

Generic data sources, direct calls, broker quantities and cash/equity balances
remain ordinary terms. Add a synonym only after a terminology decision and a
coordinated interface/documentation update.
"""

import ast
from pathlib import Path
import re

ROOT = Path(__file__).parents[1]

# These spellings describe retired framework interfaces, not arbitrary metadata.
RETIRED = (
    r"\b(?:source_keys?|PositionProposal|PositionIntent|TradeIntent|PositionState|TargetState)\b",
    r"\b(?:for_source|source_states|source_states_for_contract|expected_sources)\b",
    r"(?i:\b\w*blueprint\w*\b)",
    r"(?i:\b\w*episode\w*\b)",
    r"\bone[-_ ]to[-_ ]one\b",
    r"\b(?:BracketFutureRollExecutor|DirectFutureRollExecutor)\b",
    r"\bFutureRollMode\b|\bExecutionMode\.(?:BRACKET|DIRECT)\b",
    r"\b(?:target_)?bracket_inputs\b",
    r"\bpositions\.quantity\b",
    r"\bPortfolio\(sources\s*=",
    r"(?:__init__|Portfolio)\([^)]*\bsignals\s*=",
    r"(?i:\bdirect[- ](?:execution|targets?|rolls?|Portfolio|mode|allocation|positions?|adjustment|recovery)\b)",
    r"\b\w*(?:direct_(?:execution|target|roll|portfolio|position|mode))\w*\b",
    r"(?i:\bbracket[- ](?:rolls?|mode|family)\b)",
    r"\b\w*bracket_roll\w*\b",
    r"(?i:\bsource[- /](?:attribution|position|target|identity|exposure|allocation|key|checkpoint|quantit(?:y|ies)|universe|opt-out)\b)",
    r"(?i:\bposition[- ]balances?\b)",
    r"\b(?:position|balance|target):\{",
)
PATTERNS = tuple(re.compile(pattern) for pattern in RETIRED)

# Allow only string literals in these conversion adapters/fixtures and negative
# compatibility tests. Their implementation, identifiers and comments are checked.
LEGACY_LITERALS = {
    "scripts/migrate_components_state.py": {"_component_document", "convert_blotter"},
    "tests/test_components_migration.py": {
        "test_component_conversion_rebuilds_derived_balances_in_book",
        "keyed_target",
    },
    "tests/test_naming_schema.py": {
        "legacy_component_document",
        "test_retired_interfaces_are_absent",
        "test_retired_order_schema_never_loses_attribution",
        "test_completed_historical_roll_executor_allows_new_roll",
        "test_previous_component_conversion_preserves_reset_and_opaque_payloads",
    },
}


def checked_lines(path: Path, text: str) -> list[str]:
    """Mask only explicit old-input literals, retaining code on the same line."""
    lines = text.splitlines()
    if path.suffix != ".py":
        return lines
    literals = []
    names = LEGACY_LITERALS.get(path.relative_to(ROOT).as_posix(), set())
    for node in ast.walk(ast.parse(text)):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names
        ):
            literals.extend(
                child
                for child in ast.walk(node)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            )
        # The check's pattern declaration necessarily names retired terms.
        if path.name == "test_terminology.py" and isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "RETIRED"
                for target in node.targets
            ):
                literals.extend(
                    child
                    for child in ast.walk(node.value)
                    if isinstance(child, ast.Constant) and isinstance(child.value, str)
                )
    for literal in sorted(
        literals, key=lambda node: (node.lineno, node.col_offset), reverse=True
    ):
        for index in range(literal.lineno - 1, literal.end_lineno):
            start = literal.col_offset if index == literal.lineno - 1 else 0
            end = (
                literal.end_col_offset
                if index == literal.end_lineno - 1
                else len(lines[index])
            )
            lines[index] = (
                lines[index][:start] + " " * (end - start) + lines[index][end:]
            )
    return lines


def test_maintained_domain_terminology():
    files = {ROOT / "AGENTS.md", ROOT / "README.rst", ROOT / "pyproject.toml"}
    for directory in ("haymaker", "tests", "scripts", "docs", ".github"):
        files.update(
            path
            for path in (ROOT / directory).rglob("*")
            if path.suffix in {".py", ".rst", ".md", ".yaml", ".yml", ".toml", ".ipynb"}
            and "__pycache__" not in path.parts
        )
    violations = []
    for path in sorted(files):
        # This is the single explicit rename reference for external strategies.
        if path == ROOT / "docs/source/naming-migration.rst":
            continue
        text = path.read_text()
        original = text.splitlines()
        for number, line in enumerate(checked_lines(path, text), 1):
            if any(pattern.search(line) for pattern in PATTERNS):
                violations.append(
                    f"{path.relative_to(ROOT)}:{number}: {original[number - 1].strip()}"
                )
    assert (
        not violations
    ), "Retired terminology (see docs/source/terminology.rst):\n" + "\n".join(
        violations
    )
