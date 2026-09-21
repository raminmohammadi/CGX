"""Language-agnostic cross-file module resolution (cgx.session.cross_ref).

Gates on exactly one high-precision, un-gameable signal: a relative import that
resolves to no generated file. Everything ambiguous (assets, aliases, bare
packages, non-JS languages, paths escaping the root) must ABSTAIN so a false
red never drops a functionally-correct file.
"""

from cgx.session import cross_ref as cr


def _refs(files, paths):
    return cr.unresolved_references(files, paths, root="/proj")


def test_resolved_sibling_import_is_clean():
    files = {
        "src/App.jsx": "import { sendMessage } from './api';\n",
        "src/api.js": "export function sendMessage() {}\n",
    }
    assert _refs(files, ["src/App.jsx", "src/api.js"]) == []


def test_import_of_missing_sibling_is_flagged():
    files = {"src/App.jsx": "import { sendMessage } from './api';\n"}
    warns = _refs(files, ["src/App.jsx"])   # api.js never generated
    assert len(warns) == 1
    assert warns[0]["file"] == "src/App.jsx"
    assert warns[0]["module"] == "./api"
    assert warns[0]["kind"] == "unresolved_module"


def test_directory_index_import_resolves():
    files = {"src/main.tsx": "import App from './components';\n"}
    paths = ["src/main.tsx", "src/components/index.tsx"]
    assert _refs(files, paths) == []


def test_extensionful_relative_import_resolves():
    files = {"src/main.jsx": "import './styles-runner.js';\n"}
    assert _refs(files, ["src/main.jsx", "src/styles-runner.js"]) == []


def test_parent_dir_relative_import_resolves():
    files = {"src/pages/Home.jsx": "import { api } from '../lib/http';\n"}
    paths = ["src/pages/Home.jsx", "src/lib/http.ts"]
    assert _refs(files, paths) == []


def test_bare_thirdparty_import_is_ignored():
    files = {"src/App.jsx": "import React from 'react';\n"
                            "import axios from 'axios';\n"}
    assert _refs(files, ["src/App.jsx"]) == []


def test_alias_import_is_ignored():
    # A build-tool alias (``@/x``) is not relative -> never gated.
    files = {"src/App.jsx": "import Foo from '@/components/Foo';\n"}
    assert _refs(files, ["src/App.jsx"]) == []


def test_missing_asset_import_abstains():
    # A missing .css/.svg is ambiguous (asset may be unplanned) -> never gated.
    files = {"src/App.jsx": "import './App.css';\nimport logo from './logo.svg';\n"}
    assert _refs(files, ["src/App.jsx"]) == []


def test_missing_json_import_abstains():
    files = {"src/App.jsx": "import data from './data.json';\n"}
    assert _refs(files, ["src/App.jsx"]) == []


def test_path_escaping_root_abstains():
    # ``../../shared`` points outside the generated tree -> we cannot judge it.
    files = {"src/App.jsx": "import x from '../../shared/util';\n"}
    assert _refs(files, ["src/App.jsx"]) == []


def test_python_file_abstains():
    # Python has its own resolver; cross_ref has no row for .py.
    files = {"app.py": "from missing import thing\n"}
    assert _refs(files, ["app.py"]) == []


def test_require_and_dynamic_import_forms():
    files = {
        "src/a.js": "const b = require('./b');\n",
        "src/c.js": "const d = import('./d');\n",
    }
    warns = _refs(files, ["src/a.js", "src/c.js"])
    mods = sorted(w["module"] for w in warns)
    assert mods == ["./b", "./d"]


def test_export_from_form():
    files = {"src/index.ts": "export { X } from './x';\n"}
    assert _refs(files, ["src/index.ts"])[0]["module"] == "./x"


def test_never_raises_on_garbage():
    files = {"src/App.jsx": "\x00\xff not valid <<<", "weird": None}
    # Should not raise; garbage yields no confident unresolved import.
    assert isinstance(cr.unresolved_references(files, ["src/App.jsx"]), list)


def test_empty_inputs():
    assert cr.unresolved_references({}, []) == []
    assert cr.unresolved_references(None, None) == []
