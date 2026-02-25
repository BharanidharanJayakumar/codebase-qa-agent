import re
from collections import Counter


# Regex patterns per language to find function/class definitions
# We use named groups so the result is always {"name": ..., "type": ..., "line": ...}
SYMBOL_PATTERNS = {
    ".py": [
        r"^(?P<indent>\s*)def (?P<name>\w+)\s*\(",           # functions
        r"^(?P<indent>\s*)async def (?P<name>\w+)\s*\(",     # async functions
        r"^(?P<indent>)class (?P<name>\w+)[\s:(]",           # classes
    ],
    ".js": [
        r"function (?P<name>\w+)\s*\(",                       # named functions
        r"const (?P<name>\w+)\s*=\s*(async\s*)?\(",          # arrow functions
        r"class (?P<name>\w+)[\s{]",                          # classes
    ],
    ".ts": [
        r"function (?P<name>\w+)\s*[(<]",
        r"const (?P<name>\w+)\s*=\s*(async\s*)?\(",
        r"class (?P<name>\w+)[\s{<]",
        r"interface (?P<name>\w+)[\s{<]",
        r"type (?P<name>\w+)\s*=",
    ],
    ".go": [
        r"^func (?P<name>\w+)\s*\(",
        r"^func \(\w+ \*?\w+\) (?P<name>\w+)\s*\(",          # methods
        r"^type (?P<name>\w+) struct",
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

    Returns a list of {name, type, line} dicts.
    We use regex rather than AST parsing so this works across all languages
    without needing separate parsers installed per language.

    Trade-off: regex misses edge cases (multiline signatures, decorators),
    but it's fast, dependency-free, and good enough for our search index.
    """
    ext = "." + file_path.rsplit(".", 1)[-1] if "." in file_path else ""
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
                # Determine type from which pattern matched
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
                break  # one symbol per line max

    return symbols


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
