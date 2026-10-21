import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_VERSION = "0.6.11"
PUBLIC_FILES = (
    "README.md",
    "SECURITY.md",
    "python/README.md",
    "python/pyproject.toml",
    "typescript/README.md",
    "typescript/package.json",
)
STALE_PUBLIC_REFERENCES = ("github.com/pioneersquarelabs", "pioneer square labs")
ROOT_README_OPENING = (
    "See where your agentic AI application's LLM spend comes from. Attribute "
    "usage to functions, agents, and workflows without changing provider call "
    "sites, then send it to Metergraph Cloud or your self-hosted Metergraph "
    "server."
)


def test_public_package_files_do_not_link_to_the_previous_organization():
    for relative in PUBLIC_FILES:
        text = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8").lower()
        for stale in STALE_PUBLIC_REFERENCES:
            assert stale not in text, f"{relative} references {stale}"


def test_root_readme_opens_with_the_positioning_statement():
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    first_paragraph = readme.split("\n\n")[1]

    assert " ".join(first_paragraph.split()) == ROOT_README_OPENING


def test_python_and_typescript_package_versions_match():
    pyproject = (REPOSITORY_ROOT / "python" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    python_fallback = (
        REPOSITORY_ROOT / "python" / "src" / "metergraph" / "_version.py"
    ).read_text(encoding="utf-8")
    package = json.loads(
        (REPOSITORY_ROOT / "typescript" / "package.json").read_text(encoding="utf-8")
    )
    typescript_version = (
        REPOSITORY_ROOT / "typescript" / "src" / "version.ts"
    ).read_text(encoding="utf-8")

    assert f'version = "{EXPECTED_VERSION}"' in pyproject
    assert f'SDK_VERSION = "{EXPECTED_VERSION}"' in python_fallback
    assert package["version"] == EXPECTED_VERSION
    assert f'SDK_VERSION = "{EXPECTED_VERSION}"' in typescript_version
