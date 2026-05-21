import os
from transtructiver.parsing.annotation.builtin_checker import make_profile_from_files, is_builtin

BASE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "src", "transtructiver", "parsing", "annotation", "profiles"
)
PY = make_profile_from_files("python", os.path.join(BASE_DIR, "python"))
JAVA = make_profile_from_files("java", os.path.join(BASE_DIR, "java"))
CPP = make_profile_from_files("cpp", os.path.join(BASE_DIR, "cpp"))


def test_python_builtins():
    """Test Python builtins detection."""
    assert is_builtin("int", PY)
    assert is_builtin("len", PY)
    assert not is_builtin("b", PY)
    assert not is_builtin("my_custom_func", PY)


def test_java_builtins():
    """Test Java builtins detection."""
    assert is_builtin("String", JAVA)
    assert is_builtin("java.util", JAVA)
    assert not is_builtin("n", JAVA)
    assert not is_builtin("myHelper", JAVA)


def test_cpp_builtins():
    """Test C++ builtins detection."""
    assert not is_builtin("startswith", CPP)
    assert is_builtin("vector", CPP)
    assert not is_builtin("n", CPP)
    assert not is_builtin("my_vector", CPP)


def test_start_with_underscore():
    """Test  builtins detection for name starting with underscore."""
    assert not is_builtin("_singlelower", CPP)
    assert is_builtin("__doublelower", JAVA)
    assert is_builtin("_Singleupper", PY)


def test_qualified_names_with_reserved_fragment():
    """Qualified names with reserved fragments should be treated as builtin."""
    assert is_builtin("obj.__dunder", PY)
    assert is_builtin("module._PrivateLike", PY)
    assert is_builtin("ns::Type::__dunder", CPP)
    assert is_builtin("pkg.Type._PrivateLike", JAVA)
