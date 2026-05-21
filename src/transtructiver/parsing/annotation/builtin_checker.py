"""
Builtin Checker for Language Profiles

This module provides functions to load language builtins from JSON files and annotate AST/CST nodes
with semantic labels if their text matches any builtin identifier or type. Builtins are loaded as dictionaries
from per-language JSON files, and nodes are checked directly against keys and values (including lists).

Functions:
    load_builtins_from_json(path): Load builtins dictionary from a JSON file.
    make_profile_from_files(name, base_dir): Load builtins dict for a language from base_dir.
    normalize_name(raw_name): Strip whitespace from a name.
    is_builtin(raw_name, builtins_dict): Check if a name is a builtin (key or value).
    mark_builtin(node, builtins_dict): Annotate node with 'builtin' label if its text matches.
"""

import json
import os
import re
from collections.abc import Iterable


_LANGUAGE_KEYWORDS = {
    "python": [
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
        "self",  # not an actual keyword
    ],
    "java": [
        "abstract",
        "assert",
        "boolean",
        "break",
        "byte",
        "case",
        "catch",
        "char",
        "class",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extends",
        "final",
        "finally",
        "float",
        "for",
        "goto",
        "if",
        "implements",
        "import",
        "instanceof",
        "int",
        "interface",
        "long",
        "native",
        "new",
        "package",
        "private",
        "protected",
        "public",
        "return",
        "short",
        "static",
        "strictfp",
        "super",
        "switch",
        "synchronized",
        "this",
        "throw",
        "throws",
        "transient",
        "try",
        "void",
        "volatile",
        "while",
    ],
    "cpp": [
        "alignas",
        "alignof",
        "and",
        "and_eq",
        "asm",
        "auto",
        "bitand",
        "bitor",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "char8_t",
        "char16_t",
        "char32_t",
        "class",
        "compl",
        "concept",
        "const",
        "consteval",
        "constexpr",
        "constinit",
        "const_cast",
        "continue",
        "co_await",
        "co_return",
        "co_yield",
        "decltype",
        "default",
        "delete",
        "do",
        "double",
        "dynamic_cast",
        "else",
        "enum",
        "explicit",
        "export",
        "extern",
        "false",
        "float",
        "for",
        "friend",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "not",
        "not_eq",
        "nullptr",
        "operator",
        "or",
        "or_eq",
        "private",
        "protected",
        "public",
        "register",
        "reinterpret_cast",
        "requires",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "struct",
        "switch",
        "template",
        "this",
        "thread_local",
        "throw",
        "true",
        "try",
        "typedef",
        "typeid",
        "typename",
        "union",
        "unsigned",
        "using",
        "virtual",
        "void",
        "volatile",
        "wchar_t",
        "while",
        "xor",
        "xor_eq",
    ],
}


_LANG_VERSIONS = {
    "java": "11",
    "python": "3.8",
    "cpp": "",
}


def load_builtins_from_json(path: str) -> dict:
    """
    Load builtins dictionary from a JSON file.

    Args:
        path (str): Path to the JSON file containing builtins.

    Returns:
        dict: Dictionary of builtins (keys and lists of values).
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class ProfileDict(dict):
    """Profile dict with precomputed builtin indices for fast lookups.

    Builds two indices during initialization:
    - exact_names: set of all flat strings from the builtins dict (O(1) lookups)
    - token_index: mapping from single tokens to names containing them (fast token search)
    - _cache: LRU-style cache for is_builtin checks
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: dict[str, bool] = {}
        self._build_indices()

    def _build_indices(self) -> None:
        """Build exact_names set and token_index for fast lookups."""
        self.exact_names: set[str] = set()
        self.token_index: dict[str, set[str]] = {}

        # Collect all builtin strings from keys and values
        all_strings: list[str] = []

        # Add keys
        for key in self.keys():
            if isinstance(key, str):
                all_strings.append(key)

        # Add strings from values
        for value in self.values():
            all_strings.extend(_iter_builtin_strings(value))

        # Build exact_names set
        self.exact_names = set(all_strings)

        # Build token_index: token -> set of names containing that token
        for name in all_strings:
            tokens = _builtin_tokens(name)
            for token in tokens:
                if token not in self.token_index:
                    self.token_index[token] = set()
                self.token_index[token].add(name)

    def is_builtin_cached(self, raw_name: str) -> bool:
        """Check if name is builtin, with caching."""
        if raw_name in self._cache:
            return self._cache[raw_name]

        result = self._is_builtin_impl(raw_name)
        self._cache[raw_name] = result
        return result

    def _is_builtin_impl(self, raw_name: str) -> bool:
        """Fast builtin check using precomputed indices."""
        name = raw_name.strip()
        if not name:
            return False

        # Single-character identifiers are too noisy in profiles to treat as builtins.
        if len(name) < 3:
            return False

        # Fast path: exact match in precomputed set
        if name in self.exact_names:
            return True

        # For single-token names, check token index
        name_tokens = _builtin_tokens(name)
        if len(name_tokens) == 1:
            token = next(iter(name_tokens))
            if len(token) <= 1:
                return False
            if token in self.token_index:
                # Check if any indexed name contains this token in a matching position
                for indexed_name in self.token_index[token]:
                    if name in _builtin_tokens(indexed_name):
                        return True

        # For multi-token qualified names, require exact match (already checked above)
        return False


def make_profile_from_files(name: str, base_dir: str) -> ProfileDict:
    """
    Load builtins dictionary for a language from base_dir.

    Args:
        name (str): Language name (e.g., 'python', 'java', 'cpp').
        base_dir (str): Directory containing builtin JSON files.

    Returns:
        ProfileDict: Optimized builtins dictionary with precomputed indices.
    """
    path = os.path.join(base_dir, f"{name}{_LANG_VERSIONS[name]}_builtin_profile.json")
    builtins_dict = load_builtins_from_json(path)
    builtins_dict[f"{name}_keywords"] = _LANGUAGE_KEYWORDS[name]

    # Convert to optimized ProfileDict with precomputed indices
    profile = ProfileDict(builtins_dict)
    return profile


def is_builtin(name: str, builtins_dict: dict) -> bool:
    """
    Check if a name is a builtin.

    For ProfileDict, uses fast precomputed indices and caching.
    For plain dicts, falls back to slower exhaustive search.

    Args:
        raw_name (str): The identifier or type name to check.
        builtins_dict (dict): Builtins dictionary (preferably ProfileDict).

    Returns:
        bool: True if the name is found as a builtin, False otherwise.
    """
    if _has_reserved_identifier_fragment(name):
        return True

    if isinstance(builtins_dict, ProfileDict):
        return builtins_dict.is_builtin_cached(name)

    return False


def _has_reserved_identifier_fragment(name: str) -> bool:
    """Return True if a qualified name contains a reserved-looking identifier part.

    Examples that should match:
    - "__dunder"
    - "_PrivateLike"
    - "obj.__dunder"
    - "ns::Class::_PrivateLike"
    """
    # Split only on qualification separators; underscores are part of the identifier.
    for fragment in re.split(r"::|\.", name):
        part = fragment.strip()
        if not part:
            continue
        if part.startswith("__"):
            return True
        if part.startswith("_") and len(part) > 1 and part[1].isupper():
            return True
    return False


def _builtin_tokens(value: str) -> set[str]:
    """Split a qualified builtin name into comparable parts."""
    return {part for part in re.split(r"::|[._]+", value) if part}


def _iter_builtin_strings(value: object) -> Iterable[str]:
    """Yield all string leaves from builtin profile structures."""
    if isinstance(value, str):
        yield value
        return

    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str):
                yield key
            yield from _iter_builtin_strings(nested)
        return

    if isinstance(value, list):
        for item in value:
            yield from _iter_builtin_strings(item)
