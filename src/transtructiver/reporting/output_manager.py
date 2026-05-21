"""Output management primitives for pipeline reporting artifacts.

This module centralizes file/handle lifecycle concerns for batch processing.
It writes three artifact classes:
- Manifest records (JSONL)
- Augmented code dataset (Parquet)
- Verification summaries (CSV)

It also supports optional output sharding and gzip-compressed text outputs.
"""

import csv
import gzip
import json
import os
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..node import Node


@dataclass
class RunStats:
    """In-memory counters used to produce aggregate reporting metrics."""

    parsed_ok: int = 0
    parse_skipped: int = 0
    verified_ok: int = 0
    verified_fail: int = 0

    @property
    def processed(self) -> int:
        """Return the number of snippets that reached verification."""
        return self.verified_ok + self.verified_fail

    @property
    def success_rate(self) -> float:
        """Return the fraction of verified snippets that passed SI checks."""
        if self.processed == 0:
            return 0.0
        return self.verified_ok / self.processed


def open_text_append(path: str, compress_output: bool):
    """Open a text stream in append mode with optional gzip compression.

    Args:
        path (str): Destination file path.
        compress_output (bool): If True, open a ``.gz`` text stream.

    Returns:
        TextIO: A writable text stream opened in append mode.
    """
    if compress_output:
        return gzip.open(path, "at", encoding="utf-8", newline="")
    return open(path, "a", encoding="utf-8", newline="")


class OutputManager:
    """Manage output handles and optional shard rotation for pipeline artifacts.

    The manager is intended to be used as a context manager so all open
    resources (manifest file handles, parquet writers, summary handle) are
    deterministically closed.
    """

    def __init__(
        self,
        output_dir: str,
        max_rows_per_shard: int,
        compress_output: bool,
    ):
        """Configure output paths, shard settings, and lazily opened handles.

        Args:
            output_dir (str): Directory for output files.
            max_rows_per_shard (int): Shard size for outputs.
            compress_output (bool): Whether to gzip-compress text outputs.
        """
        self.output_dir = output_dir
        self.max_rows_per_shard = max_rows_per_shard
        self.compress_output = compress_output
        self.current_shard_id: int | None = None
        self.manifest_handle = None
        self.dataset_parquet_writer = None
        self.summary_handle = None
        self.summary_writer = None
        summary_suffix = ".csv.gz" if compress_output else ".csv"
        self.summary_path = os.path.join(output_dir, f"summary_log{summary_suffix}")

    def __enter__(self):
        """Initialize output directory and summary writer for the run.

        Returns:
            OutputManager: The initialized manager instance.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        self.summary_handle = open_text_append(self.summary_path, self.compress_output)
        self.summary_writer = csv.writer(self.summary_handle)
        return self

    def __exit__(self, exc_type, exc, tb):
        """Close all open handles/writers created during the run."""
        if self.manifest_handle is not None:
            self.manifest_handle.close()
        if self.dataset_parquet_writer is not None:
            self.dataset_parquet_writer.close()
        if self.summary_handle is not None:
            self.summary_handle.close()

    def _suffix(self, base_ext: str) -> str:
        """Return extension suffix adjusted for optional gzip text outputs.

        Args:
            base_ext (str): Base file extension.

        Returns:
            str: Extension with optional .gz suffix.
        """
        return f"{base_ext}.gz" if self.compress_output else base_ext

    def _manifest_path(self, shard_id: int) -> str:
        """Build manifest output path for a shard or single-file mode.

        Args:
            shard_id (int): Shard index.

        Returns:
            str: Manifest file path.
        """
        if self.max_rows_per_shard <= 0:
            return os.path.join(self.output_dir, f"manifest{self._suffix('.jsonl')}")
        return os.path.join(self.output_dir, f"manifest-{shard_id:06d}{self._suffix('.jsonl')}")

    def _dataset_parquet_path(self, shard_id: int) -> str:
        """Build parquet dataset output path for a shard or single-file mode.

        Args:
            shard_id (int): Shard index.

        Returns:
            str: Parquet file path.
        """
        if self.max_rows_per_shard <= 0:
            return os.path.join(self.output_dir, "augmented_dataset.parquet")
        return os.path.join(self.output_dir, f"augmented_dataset-{shard_id:06d}.parquet")

    def _ensure_shard(self, snippet_index: int, metadata: dict | None = None) -> None:
        """Rotate output handles when shard boundary changes.

        Args:
            snippet_index (int): Global snippet index used to derive shard id.
        """
        shard_id = 0
        if self.max_rows_per_shard > 0:
            shard_id = snippet_index // self.max_rows_per_shard

        if shard_id == self.current_shard_id:
            return

        if self.manifest_handle is not None:
            self.manifest_handle.close()
        if self.dataset_parquet_writer is not None:
            self.dataset_parquet_writer.close()
            self.dataset_parquet_writer = None

        manifest_path = self._manifest_path(shard_id)

        self.manifest_handle = open_text_append(manifest_path, self.compress_output)

        dataset_parquet_path = self._dataset_parquet_path(shard_id)
        schema = pa.schema(
            [
                ("snippet_id", pa.string()),
                ("original_code", pa.string()),
                ("mutated_code", pa.string()),
                ("mutated_cst", pa.string()),
                ("language", pa.string()),
                ("has_mutation_applied", pa.bool_()),
            ]
        )

        if metadata:
            schema.with_metadata(metadata)

        compression = "gzip" if self.compress_output else "snappy"
        self.dataset_parquet_writer = pq.ParquetWriter(
            dataset_parquet_path,
            schema=schema,
            compression=compression,
        )

        self.current_shard_id = shard_id

    def write_manifest(
        self, snippet_index: int, snippet_id: str, entries: list[dict[str, Any]]
    ) -> None:
        """Append one manifest record for a snippet to the current shard.

        Args:
            snippet_index (int): Index of the snippet.
            snippet_id (str): Unique snippet identifier.
            entries (list[dict[str, Any]]): Manifest entries for the snippet.
        """
        self._ensure_shard(snippet_index)
        record = {"snippet_id": snippet_id, "entries": entries}
        if self.manifest_handle is None:
            raise RuntimeError("Manifest handle is not initialized.")
        self.manifest_handle.write(json.dumps(record) + "\n")

    def write_dataset_row(
        self,
        snippet_index: int,
        snippet_id: str,
        original_code: str,
        mutated_code: str,
        language: str,
        has_mutation_applied: bool = False,
        metadata: dict | None = None,
        mutated_cst: Node | None = None,
    ) -> None:
        """Append one original/mutated code pair row to parquet output.

        Args:
            snippet_index (int): Index of the snippet.
            snippet_id (str): Unique snippet identifier.
            original_code (str): Original code string.
            mutated_code (str): Mutated code string.
            language (str): Programming language of the snippet.
            has_mutation_applied (bool): Whether any mutation was applied.
            metadata (dict | None): Additional dataset metadata (e.g., char_count, lloc, label).
            mutated_cst (Node | None): Original CST to cache for later tiers.
        """

        # Base row (core schema)
        row_dict = {
            "snippet_id": [snippet_id],
            "original_code": [original_code],
            "mutated_code": [mutated_code],
            "mutated_cst": [mutated_cst.to_json() if mutated_cst else None],
            "language": [language],
            "has_mutation_applied": [has_mutation_applied],
        }

        self._ensure_shard(snippet_index, metadata)

        if self.dataset_parquet_writer is None:
            raise RuntimeError("Parquet dataset writer is not initialized.")

        table = pa.table(row_dict, schema=self.dataset_parquet_writer.schema)

        self.dataset_parquet_writer.write_table(table)

    def output_paths_summary(self) -> tuple[str, str, str]:
        """Return human-readable paths/patterns for produced artifacts.

        Returns:
            tuple[str, str, str]: Manifest, dataset, and summary file paths.
        """
        manifest_name = "manifest-*.jsonl" if self.max_rows_per_shard > 0 else "manifest.jsonl"
        dataset_name = (
            "augmented_dataset-*.parquet"
            if self.max_rows_per_shard > 0
            else "augmented_dataset.parquet"
        )
        if self.compress_output:
            manifest_name += ".gz"
        return (
            os.path.join(self.output_dir, manifest_name),
            os.path.join(self.output_dir, dataset_name),
            self.summary_path,
        )
