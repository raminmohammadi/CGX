import ast
import logging
from typing import List, Optional

from cgx.trace import traced

logger = logging.getLogger(__name__)

class ASTAssembler:
    """Deterministically glues isolated Python AST components into a valid module.

    Takes a base string (usually the imports and globals) and parses it into
    a root ``ast.Module``. Provides methods to parse and inject additional
    functions or classes into the root module, ensuring structural validity.
    """

    def __init__(self, base_source: str = ""):
        #: Parse error of the base source, or ``None`` when it parsed.
        #: Degrading to an empty module is silent from the caller's side,
        #: and the resulting file looked like a successful (1-byte)
        #: generation; callers read this to tell the two apart.
        self.base_error: Optional[str] = None
        try:
            self.module = ast.parse(base_source)
        except SyntaxError as e:
            logger.error("Failed to parse base source for AST Assembler: %s", e)
            self.base_error = str(e)
            # Fallback to an empty module if the base source is entirely broken
            self.module = ast.parse("")

    @traced("ast_gluer.add_component")
    def add_component(self, source: str) -> bool:
        """Parse an isolated function or class and append it to the module.

        Returns True if successful, False if the injected source has a syntax error.
        """
        try:
            component_tree = ast.parse(source)
        except SyntaxError as e:
            logger.warning("AST Assembler dropped invalid component: %s", e)
            return False

        # Extend the root module's body with the top-level statements from the component
        self.module.body.extend(component_tree.body)
        return True

    @traced("ast_gluer.unparse")
    def unparse(self) -> str:
        """Convert the assembled AST back into a syntactically valid source string."""
        # ast.fix_missing_locations ensures line numbers are consistent after injection
        ast.fix_missing_locations(self.module)
        return ast.unparse(self.module)
