#!/usr/bin/env python3
"""Adopt the AX Qwen3.6 MTP sidecar into the standard MLX weight index.

The AutomatosX package stores a native-MLX-compatible ``mtp.safetensors``
next to a five-shard 6-bit target model, but its ordinary weight index does not
reference the sidecar. Standard MLX-LM therefore never opens it. This helper
creates a separate model directory, links or copies the immutable package
files, validates the exact 15-tensor native-MTP contract, and atomically writes
an index that admits the sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_TENSORS: dict[str, tuple[str, tuple[int, ...]]] = {
    "mtp.fc.weight": ("BF16", (5120, 10240)),
    "mtp.layers.0.input_layernorm.weight": ("BF16", (5120,)),
    "mtp.layers.0.mlp.down_proj.weight": ("BF16", (5120, 17408)),
    "mtp.layers.0.mlp.gate_proj.weight": ("BF16", (17408, 5120)),
    "mtp.layers.0.mlp.up_proj.weight": ("BF16", (17408, 5120)),
    "mtp.layers.0.post_attention_layernorm.weight": ("BF16", (5120,)),
    "mtp.layers.0.self_attn.k_norm.weight": ("BF16", (256,)),
    "mtp.layers.0.self_attn.k_proj.weight": ("BF16", (1024, 5120)),
    "mtp.layers.0.self_attn.o_proj.weight": ("BF16", (5120, 6144)),
    "mtp.layers.0.self_attn.q_norm.weight": ("BF16", (256,)),
    "mtp.layers.0.self_attn.q_proj.weight": ("BF16", (12288, 5120)),
    "mtp.layers.0.self_attn.v_proj.weight": ("BF16", (1024, 5120)),
    "mtp.norm.weight": ("BF16", (5120,)),
    "mtp.pre_fc_norm_embedding.weight": ("BF16", (5120,)),
    "mtp.pre_fc_norm_hidden.weight": ("BF16", (5120,)),
}

MUTABLE_NAMES = {
    "config.json",
    "model.safetensors.index.json",
    "native_mtp_adoption.json",
}
SKIP_ROOTS = {".git", ".cache"}
MLX_LM_MTP_HEAD = "a9fd7ef1032419a584ead9a38bdb66635f2d85c3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "symlink", "copy"),
        default="auto",
        help="auto tries a hardlink and falls back to an absolute symlink",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int, int]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"invalid safetensors prefix in {path}")
        header_length = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_length <= 16 * 1024 * 1024:
            raise RuntimeError(
                f"implausible safetensors header length {header_length} in {path}"
            )
        payload = handle.read(header_length)
    if len(payload) != header_length:
        raise RuntimeError(
            f"short safetensors header in {path}: {len(payload)} != {header_length}"
        )
    header = json.loads(payload)
    if not isinstance(header, dict):
        raise RuntimeError(f"invalid safetensors header object in {path}")
    tensor_rows = {key: value for key, value in header.items() if key != "__metadata__"}
    data_bytes = max(
        (int(value["data_offsets"][1]) for value in tensor_rows.values()),
        default=0,
    )
    minimum_size = 8 + header_length + data_bytes
    if path.stat().st_size < minimum_size:
        raise RuntimeError(
            f"truncated safetensors file: {path.stat().st_size} < {minimum_size}"
        )
    return tensor_rows, header_length, data_bytes


def validate_source(source_dir: Path) -> dict[str, Any]:
    required = {
        "config.json",
        "model.safetensors.index.json",
        "mtp.safetensors",
        "ax_mtp_sidecar_manifest.json",
    }
    missing = sorted(name for name in required if not (source_dir / name).is_file())
    if missing:
        raise RuntimeError(f"source package is missing: {', '.join(missing)}")

    config_path = source_dir / "config.json"
    index_path = source_dir / "model.safetensors.index.json"
    sidecar_path = source_dir / "mtp.safetensors"
    manifest_path = source_dir / "ax_mtp_sidecar_manifest.json"
    config = read_json(config_path)
    index = read_json(index_path)
    manifest = read_json(manifest_path)

    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise RuntimeError("config.json has no text_config object")
    expected_config = {
        "model_type": "qwen3_5",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 64,
        "mtp_num_hidden_layers": 1,
    }
    actual_config = {
        "model_type": config.get("model_type"),
        "hidden_size": text_config.get("hidden_size"),
        "intermediate_size": text_config.get("intermediate_size"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "mtp_num_hidden_layers": text_config.get("mtp_num_hidden_layers"),
    }
    if actual_config != expected_config:
        raise RuntimeError(
            "unexpected Qwen3.6 model contract: "
            f"expected {expected_config}, got {actual_config}"
        )
    if int(config.get("mtp_num_hidden_layers", 0) or 0) != 1:
        raise RuntimeError("top-level config does not advertise one MTP layer")

    quantization = config.get("quantization") or config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise RuntimeError("config has no quantization object")
    quant_contract = {
        "bits": int(quantization.get("bits", 0)),
        "group_size": int(quantization.get("group_size", 0)),
        "mode": quantization.get("mode"),
    }
    if quant_contract != {"bits": 6, "group_size": 64, "mode": "affine"}:
        raise RuntimeError(f"unexpected target quantization: {quant_contract}")

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("model index has no non-empty weight_map")
    existing_mtp_keys = sorted(key for key in weight_map if "mtp." in str(key))
    if existing_mtp_keys:
        raise RuntimeError(
            "source index already contains MTP keys; use that model directly instead: "
            + ", ".join(existing_mtp_keys[:5])
        )
    missing_shards = sorted(
        filename
        for filename in {str(value) for value in weight_map.values()}
        if not (source_dir / filename).is_file()
    )
    if missing_shards:
        raise RuntimeError(
            "source index references missing shard(s): " + ", ".join(missing_shards)
        )

    tensor_rows, header_length, data_bytes = read_safetensors_header(sidecar_path)
    actual_keys = set(tensor_rows)
    expected_keys = set(EXPECTED_TENSORS)
    if actual_keys != expected_keys:
        raise RuntimeError(
            "MTP sidecar tensor-key mismatch; missing="
            f"{sorted(expected_keys - actual_keys)}, extra={sorted(actual_keys - expected_keys)}"
        )
    for key, (expected_dtype, expected_shape) in EXPECTED_TENSORS.items():
        row = tensor_rows[key]
        actual_dtype = str(row.get("dtype"))
        actual_shape = tuple(int(value) for value in row.get("shape", []))
        if (actual_dtype, actual_shape) != (expected_dtype, expected_shape):
            raise RuntimeError(
                f"{key} contract mismatch: expected {(expected_dtype, expected_shape)}, "
                f"got {(actual_dtype, actual_shape)}"
            )

    manifest_mtp = manifest.get("output", {}).get("mtp", {})
    expected_sidecar_sha = str(manifest_mtp.get("sha256", ""))
    expected_sidecar_size = int(manifest_mtp.get("size_bytes", 0) or 0)
    actual_sidecar_size = sidecar_path.stat().st_size
    if expected_sidecar_size != actual_sidecar_size:
        raise RuntimeError(
            f"sidecar size does not match manifest: {actual_sidecar_size} != "
            f"{expected_sidecar_size}"
        )
    actual_sidecar_sha = sha256_file(sidecar_path)
    if not expected_sidecar_sha or actual_sidecar_sha != expected_sidecar_sha:
        raise RuntimeError(
            "sidecar SHA-256 does not match the AX provenance manifest: "
            f"{actual_sidecar_sha} != {expected_sidecar_sha or '<missing>'}"
        )

    parameter_count = sum(
        math.prod(expected_shape) for _dtype, expected_shape in EXPECTED_TENSORS.values()
    )
    return {
        "config": config,
        "index": index,
        "manifest": manifest,
        "header_length": header_length,
        "sidecar_data_bytes": data_bytes,
        "sidecar_file_bytes": actual_sidecar_size,
        "sidecar_sha256": actual_sidecar_sha,
        "sidecar_parameter_count": parameter_count,
        "source_config_sha256": sha256_file(config_path),
        "source_index_sha256": sha256_file(index_path),
        "source_manifest_sha256": sha256_file(manifest_path),
    }


def materialize_file(source: Path, destination: Path, mode: str) -> str:
    resolved = source.resolve()
    if mode in {"auto", "hardlink"}:
        try:
            os.link(resolved, destination)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    if mode in {"auto", "symlink"}:
        os.symlink(resolved, destination)
        return "symlink"
    shutil.copy2(resolved, destination)
    return "copy"


def mirror_source(source_dir: Path, stage_dir: Path, mode: str) -> Counter[str]:
    methods: Counter[str] = Counter()
    for source in sorted(source_dir.rglob("*")):
        relative = source.relative_to(source_dir)
        if relative.parts and relative.parts[0] in SKIP_ROOTS:
            continue
        destination = stage_dir / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if relative.as_posix() in MUTABLE_NAMES:
            if relative.name == "native_mtp_adoption.json":
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.resolve(), destination)
            methods["copy-mutable"] += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        methods[materialize_file(source, destination, mode)] += 1
    return methods


def build_adopted_model(
    source_dir: Path,
    output_dir: Path,
    link_mode: str,
) -> dict[str, Any]:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().absolute()
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"output path already exists: {output_dir}; choose a new native-MTP directory"
        )
    if output_dir == source_dir or output_dir.is_relative_to(source_dir):
        raise RuntimeError("output directory must not be the source or live inside it")

    source_contract = validate_source(source_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    stage_dir.mkdir()
    try:
        methods = mirror_source(source_dir, stage_dir, link_mode)
        index_path = stage_dir / "model.safetensors.index.json"
        index = read_json(index_path)
        weight_map = index["weight_map"]
        for key in sorted(EXPECTED_TENSORS):
            weight_map[key] = "mtp.safetensors"

        metadata = index.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        base_total_size = int(metadata.get("total_size", 0) or 0)
        metadata.update(
            {
                "total_size": base_total_size
                + int(source_contract["sidecar_data_bytes"]),
                "mtp_tensor_count": len(EXPECTED_TENSORS),
                "mtp_tensor_bytes": int(source_contract["sidecar_data_bytes"]),
                "mtp_parameters": int(source_contract["sidecar_parameter_count"]),
                "native_mtp_contract": "mlx-lm#1740",
            }
        )
        index["metadata"] = metadata
        write_json(index_path, index)

        receipt = {
            "schema": "vllm-metal.ax-qwen36-native-mtp-adoption.v1",
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "requested_link_mode": link_mode,
            "materialization_methods": dict(sorted(methods.items())),
            "mlx_lm_mtp_head": MLX_LM_MTP_HEAD,
            "source_config_sha256": source_contract["source_config_sha256"],
            "source_index_sha256": source_contract["source_index_sha256"],
            "source_manifest_sha256": source_contract["source_manifest_sha256"],
            "sidecar_sha256": source_contract["sidecar_sha256"],
            "sidecar_file_bytes": source_contract["sidecar_file_bytes"],
            "sidecar_data_bytes": source_contract["sidecar_data_bytes"],
            "sidecar_parameter_count": source_contract["sidecar_parameter_count"],
            "mtp_tensor_count": len(EXPECTED_TENSORS),
            "mtp_tensor_keys": sorted(EXPECTED_TENSORS),
            "patched_index_sha256": sha256_file(index_path),
        }
        write_json(stage_dir / "native_mtp_adoption.json", receipt)

        # Re-read the staged result before making it visible.
        staged_index = read_json(index_path)
        staged_map = staged_index.get("weight_map", {})
        if any(staged_map.get(key) != "mtp.safetensors" for key in EXPECTED_TENSORS):
            raise RuntimeError("staged index failed its MTP mapping verification")
        if sha256_file(stage_dir / "mtp.safetensors") != receipt["sidecar_sha256"]:
            raise RuntimeError("staged sidecar digest changed during materialization")
        os.replace(stage_dir, output_dir)
    except BaseException:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    return receipt


def main() -> int:
    args = parse_args()
    receipt = build_adopted_model(args.source_dir, args.output_dir, args.link_mode)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"Native-MTP model directory created: {args.output_dir.expanduser().absolute()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
