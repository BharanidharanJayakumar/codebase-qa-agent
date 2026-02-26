"""
Tree-sitter AST parsing for accurate symbol extraction.
Falls back to regex (extractor.py) if tree-sitter grammars aren't installed.

Grammars are pip-installed per language:
  pip install tree-sitter-python tree-sitter-javascript tree-sitter-typescript ...
"""
from pathlib import Path

# Map file extensions to tree-sitter language modules
_LANG_MODULES = {
    ".py": "tree_sitter_python",
    ".js": "tree_sitter_javascript",
    ".jsx": "tree_sitter_javascript",
    ".ts": "tree_sitter_typescript",
    ".tsx": "tree_sitter_typescript",
    ".go": "tree_sitter_go",
    ".java": "tree_sitter_java",
    ".rs": "tree_sitter_rust",
    ".rb": "tree_sitter_ruby",
    ".cs": "tree_sitter_c_sharp",
    ".c": "tree_sitter_c",
    ".cpp": "tree_sitter_cpp",
    ".php": "tree_sitter_php",
}

# Tree-sitter node types that represent symbols we care about
_SYMBOL_NODE_TYPES = {
    # Python
    "function_definition": "function",
    "class_definition": "class",
    # JavaScript / TypeScript
    "function_declaration": "function",
    "class_declaration": "class",
    "method_definition": "function",
    "arrow_function": "function",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    # Go
    "function_declaration": "function",
    "method_declaration": "function",
    "type_declaration": "type",
    # Java / C#
    "method_declaration": "function",
    "constructor_declaration": "function",
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "class",
    # Rust
    "function_item": "function",
    "struct_item": "class",
    "trait_item": "interface",
    "impl_item": "class",
    # Ruby
    "method": "function",
    "class": "class",
    "module": "class",
    # C/C++
    "function_definition": "function",
    "struct_specifier": "class",
    "class_specifier": "class",
}

_loaded_languages = {}


def _load_language(ext: str):
    """Try to load a tree-sitter language for the given extension."""
    module_name = _LANG_MODULES.get(ext)
    if not module_name:
        return None

    if module_name in _loaded_languages:
        return _loaded_languages[module_name]

    try:
        import importlib
        import tree_sitter
        lang_mod = importlib.import_module(module_name)
        language = tree_sitter.Language(lang_mod.language())
        _loaded_languages[module_name] = language
        return language
    except (ImportError, AttributeError, Exception):
        _loaded_languages[module_name] = None
        return None


def extract_symbols_ast(content: str, file_path: str) -> list[dict] | None:
    """
    Extract symbols using tree-sitter AST parsing.
    Returns None if tree-sitter is not available for this language (caller should fall back to regex).
    Returns list of {name, type, line} on success.
    """
    ext = Path(file_path).suffix
    language = _load_language(ext)
    if language is None:
        return None  # Signal caller to use regex fallback

    try:
        import tree_sitter
    except ImportError:
        return None

    parser = tree_sitter.Parser(language)
    tree = parser.parse(content.encode("utf-8"))

    symbols = []
    _walk_tree(tree.root_node, symbols, content)
    return symbols


def _walk_tree(node, symbols: list, content: str) -> None:
    """Recursively walk the AST and collect symbol definitions."""
    node_type = node.type

    if node_type in _SYMBOL_NODE_TYPES:
        name = _extract_name(node, content)
        if name:
            symbols.append({
                "name": name,
                "type": _SYMBOL_NODE_TYPES[node_type],
                "line": node.start_point[0] + 1,  # tree-sitter is 0-indexed
            })

    for child in node.children:
        _walk_tree(child, symbols, content)


def _extract_name(node, content: str) -> str | None:
    """Extract the name identifier from a symbol node."""
    # Most nodes have a direct 'name' child
    for child in node.children:
        if child.type in ("identifier", "property_identifier", "type_identifier"):
            return content[child.start_byte:child.end_byte]

    # For arrow functions assigned to variables: const foo = () => {}
    if node.type == "arrow_function" and node.parent:
        parent = node.parent
        if parent.type in ("variable_declarator", "assignment_expression"):
            for child in parent.children:
                if child.type in ("identifier", "property_identifier"):
                    return content[child.start_byte:child.end_byte]

    return None


def is_available(ext: str) -> bool:
    """Check if tree-sitter parsing is available for this file extension."""
    return _load_language(ext) is not None
