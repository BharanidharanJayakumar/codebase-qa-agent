import re
from collections import Counter
from pathlib import Path


# Regex patterns per language to find function/class definitions
# We use named groups so the result is always {"name": ..., "type": ..., "line": ...}
SYMBOL_PATTERNS = {
    # --- Python ---
    ".py": [
        r"^(?P<indent>\s*)def (?P<name>\w+)\s*\(",           # functions
        r"^(?P<indent>\s*)async def (?P<name>\w+)\s*\(",     # async functions
        r"^(?P<indent>)class (?P<name>\w+)[\s:(]",           # classes
    ],

    # --- JavaScript ---
    ".js": [
        r"function (?P<name>\w+)\s*\(",                       # named functions
        r"const (?P<name>\w+)\s*=\s*(async\s*)?\(",          # arrow functions
        r"class (?P<name>\w+)[\s{]",                          # classes
    ],
    ".jsx": [
        r"function (?P<name>\w+)\s*\(",
        r"const (?P<name>\w+)\s*=\s*(async\s*)?\(",
        r"class (?P<name>\w+)[\s{]",
    ],

    # --- TypeScript ---
    ".ts": [
        r"function (?P<name>\w+)\s*[(<]",
        r"const (?P<name>\w+)\s*=\s*(async\s*)?\(",
        r"class (?P<name>\w+)[\s{<]",
        r"interface (?P<name>\w+)[\s{<]",
        r"type (?P<name>\w+)\s*=",
    ],
    ".tsx": [
        r"function (?P<name>\w+)\s*[(<]",
        r"const (?P<name>\w+)\s*=\s*(async\s*)?\(",
        r"class (?P<name>\w+)[\s{<]",
        r"interface (?P<name>\w+)[\s{<]",
    ],

    # --- Go ---
    ".go": [
        r"^func (?P<name>\w+)\s*\(",                          # top-level functions
        r"^func \(\w+ \*?\w+\) (?P<name>\w+)\s*\(",          # methods on types
        r"^type (?P<name>\w+) struct",                        # structs
        r"^type (?P<name>\w+) interface",                     # interfaces
    ],

    # --- Java ---
    # Covers: public/private/protected methods, classes, interfaces, enums, records (Java 16+)
    # Known gap: complex generics like List<Map<K,V>> are partially matched — good enough for search
    ".java": [
        r"(public|private|protected|static|final|\s)+[\w<>\[\]]+\s+(?P<name>\w+)\s*\(",  # methods
        r"\bclass\s+(?P<name>\w+)[\s{<]",                    # classes
        r"\binterface\s+(?P<name>\w+)[\s{<]",                # interfaces
        r"\benum\s+(?P<name>\w+)[\s{]",                      # enums
        r"\brecord\s+(?P<name>\w+)\s*[\s({]",                # records (Java 16+)
    ],

    # --- C# / .NET ---
    # Covers: methods, classes, interfaces, enums, records, delegates
    # Supports modern C# patterns: record types, init-only properties
    ".cs": [
        r"(public|private|protected|internal|static|virtual|override|abstract|\s)+[\w<>\[\]?]+\s+(?P<name>\w+)\s*\(",  # methods
        r"\bclass\s+(?P<name>\w+)[\s:{<]",                   # classes
        r"\binterface\s+(?P<name>\w+)[\s:{<]",               # interfaces
        r"\benum\s+(?P<name>\w+)[\s{]",                      # enums
        r"\brecord\s+(?P<name>\w+)[\s({]",                   # records (C# 9+)
        r"\bdelegate\s+[\w<>\[\]]+\s+(?P<name>\w+)\s*\(",   # delegates
    ],

    # --- Rust ---
    ".rs": [
        r"^pub\s+(async\s+)?fn\s+(?P<name>\w+)\s*[(<]",     # public functions
        r"^(async\s+)?fn\s+(?P<name>\w+)\s*[(<]",           # private functions
        r"^pub\s+struct\s+(?P<name>\w+)[\s{<]",             # public structs
        r"^struct\s+(?P<name>\w+)[\s{<]",                    # private structs
        r"^pub\s+trait\s+(?P<name>\w+)[\s{<]",              # traits
        r"^pub\s+enum\s+(?P<name>\w+)[\s{<]",               # enums
        r"^impl\s+(?P<name>\w+)[\s{<]",                     # impl blocks
    ],

    # --- Ruby ---
    ".rb": [
        r"^\s*def\s+(?P<name>\w+[\?!]?)",                    # methods (including ? and ! variants)
        r"^\s*class\s+(?P<name>\w+)",                        # classes
        r"^\s*module\s+(?P<name>\w+)",                       # modules
    ],

    # --- PHP ---
    ".php": [
        r"function\s+(?P<name>\w+)\s*\(",                    # functions
        r"class\s+(?P<name>\w+)[\s{]",                       # classes
        r"interface\s+(?P<name>\w+)[\s{]",                   # interfaces
        r"trait\s+(?P<name>\w+)[\s{]",                       # traits
    ],

    # --- C / C++ ---
    ".c": [
        r"^[\w\s\*]+\s+(?P<name>\w+)\s*\([^;]",             # function definitions (not declarations)
    ],
    ".cpp": [
        r"^[\w\s\*:<>]+\s+(?P<name>\w+)\s*\([^;]",          # functions and methods
        r"\bclass\s+(?P<name>\w+)[\s:{]",                    # classes
        r"\bstruct\s+(?P<name>\w+)[\s:{]",                   # structs
    ],
}

# Words that appear everywhere and carry no meaning for search
STOP_WORDS = {
    "the", "a", "an", "is", "in", "it", "of", "to", "and", "or",
    "for", "with", "this", "that", "be", "are", "was", "were",
    "import", "from", "return", "if", "else", "elif", "class",
    "def", "function", "const", "let", "var", "true", "false",
    "none", "null", "self", "type", "pass", "print",
}


def extract_symbols(content: str, file_path: str) -> list[dict]:
    """
    Extract function, class, and type definitions from source code.
    Tries tree-sitter AST parsing first (accurate), falls back to regex (fast, universal).
    Returns a list of {name, type, line} dicts.
    """
    # Try tree-sitter first for accurate parsing
    try:
        from skills.ast_parser import extract_symbols_ast
        ast_result = extract_symbols_ast(content, file_path)
        if ast_result is not None:
            return ast_result
    except ImportError:
        pass

    # Fall back to regex
    return _extract_symbols_regex(content, file_path)


def _extract_symbols_regex(content: str, file_path: str) -> list[dict]:
    """Regex-based symbol extraction. Works across all languages without dependencies."""
    ext = Path(file_path).suffix
    patterns = SYMBOL_PATTERNS.get(ext, [])
    if not patterns:
        return []

    symbols = []
    lines = content.splitlines()

    for line_num, line in enumerate(lines, start=1):
        for pattern in patterns:
            match = re.search(pattern, line)
            if match and "name" in match.groupdict():
                name = match.group("name")
                symbol_type = (
                    "class" if "class" in pattern
                    else "interface" if "interface" in pattern
                    else "type" if "type" in pattern
                    else "function"
                )
                symbols.append({
                    "name": name,
                    "type": symbol_type,
                    "line": line_num,
                })
                break
    return symbols


def chunk_file(content: str, symbols: list[dict], max_chunk_lines: int = 60) -> list[dict]:
    """
    Split file content into semantic chunks at symbol boundaries.

    Instead of storing the first 4000 chars (losing everything after line ~100),
    we split at function/class boundaries so every symbol body is preserved.

    Each chunk has: start_line, end_line, content, symbol (if the chunk starts at one).
    Files with no symbols get a single full-file chunk.
    """
    lines = content.splitlines()
    total_lines = len(lines)

    if not symbols or total_lines == 0:
        # No symbols detected — store the whole file as one chunk (up to 200 lines)
        return [{"start_line": 1, "end_line": min(total_lines, 200),
                 "content": "\n".join(lines[:200]), "symbol": None}]

    # Sort symbols by line number
    sorted_syms = sorted(symbols, key=lambda s: s["line"])

    chunks = []

    # If the file starts with content before the first symbol (imports, comments),
    # capture that as a header chunk
    first_sym_line = sorted_syms[0]["line"]
    if first_sym_line > 1:
        header_end = first_sym_line - 1
        chunks.append({
            "start_line": 1,
            "end_line": header_end,
            "content": "\n".join(lines[:header_end]),
            "symbol": None,
        })

    # Create a chunk for each symbol — from its line to the next symbol (or EOF)
    for i, sym in enumerate(sorted_syms):
        start = sym["line"]
        if i + 1 < len(sorted_syms):
            end = sorted_syms[i + 1]["line"] - 1
        else:
            end = total_lines

        # Cap at max_chunk_lines to avoid giant chunks
        end = min(end, start + max_chunk_lines - 1)

        chunk_lines = lines[start - 1:end]  # lines is 0-indexed, symbols are 1-indexed
        chunks.append({
            "start_line": start,
            "end_line": end,
            "content": "\n".join(chunk_lines),
            "symbol": sym["name"],
        })

    return chunks


def extract_keywords(content: str, top_n: int = 20) -> list[str]:
    """
    Extract the most meaningful words from a file's content.

    This builds the keyword_map in memory — so when you search for "auth"
    we know which files contain that concept without re-reading anything.

    Approach: simple word frequency after filtering stop words and short tokens.
    No embeddings, no vector DB — fast and works offline with zero dependencies.
    """
    # Strip code syntax: keep only alphabetic words of length >= 3
    words = re.findall(r"[a-zA-Z]{3,}", content.lower())

    # Split camelCase and snake_case identifiers into sub-words
    # e.g. "getUserById" → ["get", "user", "by", "id"]
    expanded = []
    for word in words:
        # Split camelCase
        sub = re.sub(r"([a-z])([A-Z])", r"\1 \2", word).lower().split()
        # Split snake_case
        for part in sub:
            expanded.extend(part.split("_"))

    # Filter stop words and very short tokens
    meaningful = [w for w in expanded if w not in STOP_WORDS and len(w) > 2]

    # Return top_n most frequent — these represent what the file is "about"
    counter = Counter(meaningful)
    return [word for word, _ in counter.most_common(top_n)]
