import importlib
import json

import pytest


def test_runstats_success_rate():
    """Test RunStats processed and success_rate properties."""
    module = importlib.import_module("transtructiver.reporting.output_manager")

    stats = module.RunStats(parsed_ok=3, parse_skipped=1, verified_ok=2, verified_fail=2)

    assert stats.processed == 4
    assert stats.success_rate == 0.5


def test_output_paths_summary_single_file(tmp_path):
    """Test output_paths_summary returns correct file names for single file mode."""
    module = importlib.import_module("transtructiver.reporting.output_manager")

    manager = module.OutputManager(str(tmp_path), max_rows_per_shard=0, compress_output=False)
    manifest_path, dataset_path, summary_path = manager.output_paths_summary()

    assert manifest_path.endswith("manifest.jsonl")
    assert dataset_path.endswith("augmented_dataset.parquet")
    assert summary_path.endswith("summary_log.csv")


def test_output_manager_writes_manifest_and_dataset(tmp_path):
    """Test OutputManager writes manifest and dataset files correctly."""
    module = importlib.import_module("transtructiver.reporting.output_manager")

    with module.OutputManager(
        str(tmp_path), max_rows_per_shard=0, compress_output=False
    ) as manager:
        manager.write_manifest(0, "row_0", [{"node_id": [0, 0], "history": [], "metadata": {}}])
        manager.write_dataset_row(0, "row_0", "original", "mutated", "python")

    manifest_file = tmp_path / "manifest.jsonl"
    dataset_file = tmp_path / "augmented_dataset.parquet"
    summary_file = tmp_path / "summary_log.csv"

    assert manifest_file.exists()
    assert dataset_file.exists()
    assert summary_file.exists()

    payload = json.loads(manifest_file.read_text(encoding="utf-8").strip())
    assert payload["snippet_id"] == "row_0"
    assert payload["entries"][0]["node_id"] == [0, 0]


def test_output_manager_writes_mutated_cst(tmp_path):
    """Test OutputManager stores mutated CST in the dataset output."""
    module = importlib.import_module("transtructiver.reporting.output_manager")
    node_module = importlib.import_module("transtructiver.node")
    parquet = importlib.import_module("pyarrow.parquet")

    mutated_cst = node_module.Node((0, 0), (0, 1), "identifier", text="x")

    with module.OutputManager(
        str(tmp_path), max_rows_per_shard=0, compress_output=False
    ) as manager:
        manager.write_dataset_row(
            0,
            "row_0",
            "original",
            "mutated",
            "python",
            mutated_cst=mutated_cst,
        )

    dataset_file = tmp_path / "augmented_dataset.parquet"
    table = parquet.read_table(dataset_file)
    data = table.to_pydict()

    assert data["mutated_cst"][0] == mutated_cst.to_json()
