import importlib
import json

import pytest


def test_validate_rules_flags_unsupported():
    """Test validate_rules flags unsupported rule names."""
    module = importlib.import_module("transtructiver.cli")

    unsupported = module._validate_rules(["rename-identifier", "does-not-exist"])

    assert unsupported == ["does-not-exist"]


def test_run_pipeline_happy_path_writes_artifacts_and_summary(monkeypatch, tmp_path):
    """Test run_pipeline writes artifacts and summary in happy path."""
    module = importlib.import_module("transtructiver.cli")

    class DummyLoader:
        def __init__(self, *args, **kwargs):
            self.num_rows = 0
            pass

        def load_checkpoint(self, resume):
            return 0

        def save_checkpoint(self, next_index, stats):
            save_calls.append((next_index, stats.parsed_ok, stats.verified_ok, stats.verified_fail))

        def iter_snippets(self, batch_size, start_index):
            return [(0, {"snippet_id": "row_0", "code": "print(1)", "language": "python"})]

    class DummyNode:
        def __init__(self, code):
            self.code = code

        def clone(self):
            return DummyNode(self.code)

        def to_code(self):
            return self.code

    class FakeParser:
        def parse(self, code, language):
            return DummyNode("original_code"), None

    class FakeManifest:
        def to_dict(self):
            return [{"node_id": [0, 0], "history": [], "metadata": {}}]

        def is_empty(self):
            # Return True when manifest has no entries
            return len(self.to_dict()) == 0

    class FakeEngine:
        def __init__(self):
            self.manifest = FakeManifest()

        def apply_mutations(self, node):
            node.code = "mutated_code"

    class FakeVerifier:
        def __init__(self):
            self.errors = []

        def verify(self, orig, mut, manifest):
            return True

    class FakeOutputManager:
        instance = None

        def __init__(self, *args, **kwargs):
            self.summary_writer = object()
            self.manifest_calls = []
            self.dataset_calls = []
            FakeOutputManager.instance = self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write_manifest(self, idx, snippet_id, entries):
            self.manifest_calls.append((idx, snippet_id, entries))

        def write_dataset_row(
            self,
            snippet_idx,
            snippet_id,
            original_code,
            mutated_code,
            language,
            has_mutation_applied,
            metadata,
            mutated_cst=None,
        ):
            metadata = {
                "char_count": 1,
                "loc": 1,
                "lloc": 1,
                "for_loop_density": 0.1,
                "identifier_density": 0.1,
                "comment_density": 0.1,
                "whitespace_ratio": 0.1,
                "code_hash": "x",
                "label": "y",
            }

            self.dataset_calls.append(
                (
                    snippet_idx,
                    snippet_id,
                    original_code,
                    mutated_code,
                    language,
                    has_mutation_applied,
                    metadata,
                    mutated_cst,
                )
            )

        def output_paths_summary(self):
            return ("manifest.jsonl", "augmented_dataset.parquet", "summary_log.csv")

    summary_calls = []
    totals_calls = []
    save_calls = []

    def fake_write_summary(**kwargs):
        summary_calls.append(kwargs)

    def fake_write_summary_totals(**kwargs):
        totals_calls.append(kwargs)

    monkeypatch.setattr(module, "Parser", FakeParser)
    monkeypatch.setattr(module, "_build_engine", lambda *args, **kwargs: FakeEngine())
    monkeypatch.setattr(module, "SIVerifier", FakeVerifier)
    monkeypatch.setattr(module, "OutputManager", FakeOutputManager)
    monkeypatch.setattr(module.summary_logger, "write_summary", fake_write_summary)
    monkeypatch.setattr(module.summary_logger, "write_summary_totals", fake_write_summary_totals)
    monkeypatch.setattr(module, "_prototype_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_prototype_pretty", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "DataLoader", lambda *args, **kwargs: DummyLoader())

    checkpoint_path = tmp_path / "checkpoint.json"
    options = module.PipelineOptions(checkpoint_path=str(checkpoint_path), checkpoint_every=1000)

    module.run_pipeline(
        filepath="dummy.parquet",
        rules=["rename-identifier"],
        output_dir=str(tmp_path),
        rule_params={"level": 1, "targets": ["variable"]},
        pipeline_options=options,
    )

    manager = FakeOutputManager.instance
    assert manager is not None
    assert len(manager.manifest_calls) == 1
    assert len(manager.dataset_calls) == 1

    assert summary_calls[0]["snippet_id"] == "row_0"
    assert summary_calls[0]["verified"] is True
    assert summary_calls[0]["writer"] is manager.summary_writer

    assert totals_calls[0]["parsed_ok"] == 1
    assert totals_calls[0]["verified_ok"] == 1
    assert totals_calls[0]["verified_fail"] == 0

    assert len(save_calls) == 1
    assert save_calls[0][1] == 1


def test_run_pipeline_raises_for_unsupported_rule(tmp_path):
    """Test run_pipeline raises ValueError for unsupported rule."""
    module = importlib.import_module("transtructiver.cli")

    with pytest.raises(ValueError):
        module.run_pipeline(
            filepath="dummy.parquet",
            rules=["unsupported-rule"],
            output_dir=str(tmp_path),
        )


def test_run_pipeline_integration_writes_real_outputs(monkeypatch, tmp_path):
    """Test run_pipeline integration writes real output files."""
    module = importlib.import_module("transtructiver.cli")

    class DummyLoader:
        def __init__(self, *args, **kwargs):
            self.checkpoint_path = str(checkpoint_path)
            self.num_rows = 0

        def load_checkpoint(self, resume):
            return 0

        def save_checkpoint(self, next_index, stats):
            # Actually write a checkpoint file so the test can check for it
            payload = {"next_index": next_index}
            with open(self.checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

        def iter_snippets(self, batch_size, start_index):
            return [(0, {"snippet_id": "row_0", "code": snippet, "language": "python"})]

    snippet = "def add(a, b):\n    return a + b\n"
    monkeypatch.setattr(module, "_prototype_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_prototype_pretty", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "DataLoader", lambda *args, **kwargs: DummyLoader())

    output_dir = tmp_path / "out"
    checkpoint_path = tmp_path / "checkpoint.json"
    options = module.PipelineOptions(checkpoint_path=str(checkpoint_path), checkpoint_every=1000)

    module.run_pipeline(
        filepath="ignored.parquet",
        rules=["rename-identifier"],
        output_dir=str(output_dir),
        pipeline_options=options,
    )

    manifest_file = output_dir / "manifest.jsonl"
    dataset_file = output_dir / "augmented_dataset.parquet"
    summary_file = output_dir / "summary_log.csv"

    assert manifest_file.exists()
    assert dataset_file.exists()
    assert summary_file.exists()
    assert checkpoint_path.exists()

    manifest_row = json.loads(manifest_file.read_text(encoding="utf-8").splitlines()[0])
    assert manifest_row["snippet_id"] == "row_0"

    summary_rows = summary_file.read_text(encoding="utf-8").splitlines()
    assert summary_rows[0].startswith("row_0,")
    assert summary_rows[1].startswith("TOTAL,")


def test_run_pipeline_use_cached_original_cst_on_mutated_code(monkeypatch, tmp_path):
    module = importlib.import_module("transtructiver.cli")

    class DummyLoader:
        def __init__(self, *args, **kwargs):
            self.checkpoint_path = str(checkpoint_path)
            self.num_rows = 0

        def load_checkpoint(self, resume):
            return 0

        def save_checkpoint(self, next_index, stats):
            # Actually write a checkpoint file so the test can check for it
            payload = {"next_index": next_index}
            with open(self.checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

        def iter_snippets(self, batch_size, start_index):
            return [
                (
                    0,
                    {
                        "snippet_id": "row_0",
                        "original_code": snippet,
                        "mutated_code": mut_snippet,
                        "language": "python",
                        "mutated_cst": json_cst,
                    },
                )
            ]

    snippet = "def add(a, b):\n    return a + b\n"
    mut_snippet = "def add_func(a_var, b_var):\n    return a_var + b_var\n"
    orig_cst = {
        "start_point": (0, 0),
        "end_point": (0, len(snippet)),
        "type": "module",
        "text": snippet,
        "semantic_label": "root",
        "context_type": None,
        "field": None,
        "language": "python",
        "builtin": False,
        "is_named": True,
        "children": [],
    }
    json_cst = json.dumps(orig_cst, ensure_ascii=False)
    monkeypatch.setattr(module, "_prototype_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_prototype_pretty", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "DataLoader", lambda *args, **kwargs: DummyLoader())

    output_dir = tmp_path / "out"
    checkpoint_path = tmp_path / "checkpoint.json"
    options = module.PipelineOptions(checkpoint_path=str(checkpoint_path), checkpoint_every=1000)

    module.run_pipeline(
        filepath="ignored.parquet",
        rules=["whitespace-normalization"],
        output_dir=str(output_dir),
        pipeline_options=options,
    )

    manifest_file = output_dir / "manifest.jsonl"
    dataset_file = output_dir / "augmented_dataset.parquet"
    summary_file = output_dir / "summary_log.csv"

    assert manifest_file.exists()
    assert dataset_file.exists()
    assert summary_file.exists()
    assert checkpoint_path.exists()

    manifest_row = json.loads(manifest_file.read_text(encoding="utf-8").splitlines()[0])
    assert manifest_row["snippet_id"] == "row_0"

    summary_rows = summary_file.read_text(encoding="utf-8").splitlines()
    assert summary_rows[0].startswith("row_0,")
    assert summary_rows[1].startswith("TOTAL,")
