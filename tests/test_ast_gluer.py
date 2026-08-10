import ast
import pytest
from cgx.codegen.ast_gluer import ASTAssembler

def test_ast_assembler_basic():
    header = "import os\nimport sys\n\nGLOBAL_VAR = 42\n"
    assembler = ASTAssembler(header)
    
    # Verify header is correctly parsed
    assert isinstance(assembler.module, ast.Module)
    
    func_source = "def my_func():\n    return GLOBAL_VAR\n"
    success = assembler.add_component(func_source)
    assert success is True
    
    final_code = assembler.unparse()
    assert "import os" in final_code
    assert "def my_func():" in final_code
    assert "return GLOBAL_VAR" in final_code

def test_ast_assembler_syntax_error():
    header = "import os\n"
    assembler = ASTAssembler(header)
    
    # Function missing colon
    bad_func = "def bad_func()\n    pass"
    success = assembler.add_component(bad_func)
    
    # Should safely reject bad syntax
    assert success is False
    
    final_code = assembler.unparse()
    assert "bad_func" not in final_code
    assert "import os" in final_code

def test_ast_assembler_bad_header():
    # If the header itself is a syntax error, it should degrade gracefully
    header = "import os\nbad syntax here"
    assembler = ASTAssembler(header)
    
    func_source = "def ok_func():\n    pass"
    success = assembler.add_component(func_source)
    assert success is True
    
    final_code = assembler.unparse()
    assert "def ok_func():" in final_code
    assert "bad syntax here" not in final_code
    # The degradation must be visible: a caller that cannot tell an empty
    # module from a parsed one shipped a 1-byte file as a success.
    assert assembler.base_error


def test_ast_assembler_reports_no_error_for_a_valid_header():
    assert ASTAssembler("import os\n").base_error is None
    assert ASTAssembler("").base_error is None
