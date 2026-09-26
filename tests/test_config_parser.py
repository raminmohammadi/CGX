"""Config / CI / infra file ingestion (cgx.parser.config_parser).

Guards the coverage gap where CI/config files (bitbucket-pipelines.yml,
.github/workflows/*, Dockerfile, ...) were never indexed, so "is there a CI
pipeline?" came back empty even with the file in the repo.
"""

from __future__ import annotations

import textwrap

from cgx.parser.config_parser import ConfigParser
from cgx.parser.parse_codebase import _PARSER_REGISTRY, parse_codebase


def test_config_parser_emits_searchable_doc_chunk():
    src = textwrap.dedent(
        """
        image: node:20
        pipelines:
          default:
            - step:
                script: [npm ci, npm test, npm run build]
        """
    )
    chunks, calls = ConfigParser().parse_file("repo/bitbucket-pipelines.yml", src, "repo")
    assert calls == []
    assert len(chunks) == 1
    c = chunks[0]
    assert c["type"] == "doc"
    assert c["name"] == "bitbucket-pipelines.yml"
    assert c["meta"]["source_kind"] == "doc"          # flows through as prose
    assert c["meta"]["config_kind"] == "ci"
    assert "pipelines" in c["code"] and "npm test" in c["code"]  # body is searchable


def test_dockerfile_matched_by_basename_and_labelled():
    chunks, _ = ConfigParser().parse_file("repo/Dockerfile", "FROM python:3.12\nRUN pip install x\n", "repo")
    assert len(chunks) == 1 and chunks[0]["meta"]["config_kind"] == "docker"


def test_lockfiles_are_skipped():
    for name in ("package-lock.json", "uv.lock", "yarn.lock"):
        assert ConfigParser().parse_file(f"repo/{name}", '{"a": 1}\n', "repo") == ([], [])


def test_registry_covers_config_extensions_and_basenames():
    for key in (".yml", ".yaml", ".toml", ".json", "Dockerfile", "Makefile"):
        assert key in _PARSER_REGISTRY, f"{key} not registered"


def test_parse_codebase_ingests_config_and_skips_lockfiles(tmp_path):
    (tmp_path / "bitbucket-pipelines.yml").write_text(
        "pipelines:\n  default:\n    - step:\n        script: [make deploy]\n")
    (tmp_path / "Dockerfile").write_text("FROM alpine\nCMD sh\n")
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 3}\n')
    (tmp_path / "app.py").write_text('"""m."""\n\ndef f():\n    return 1\n')
    chunks, _ = parse_codebase(str(tmp_path))
    names = {c["name"] for c in chunks if c.get("meta", {}).get("config_kind")}
    assert "bitbucket-pipelines.yml" in names
    assert "Dockerfile" in names
    assert "package-lock.json" not in names  # lockfile skipped
