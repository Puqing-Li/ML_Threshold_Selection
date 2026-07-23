"""Validation helpers for explicit, sample-specific voxel sizes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math


FEATURE_SCHEMA_VERSION = "resolution_aware_v2_per_sample_sqrt5"


def parse_voxel_size_mm(value, sample_id: str) -> float:
    """Return one finite, positive voxel edge length in mm."""
    try:
        voxel_size_mm = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Voxel size for sample '{sample_id}' must be a number in mm/voxel."
        ) from exc
    if not math.isfinite(voxel_size_mm) or voxel_size_mm <= 0:
        raise ValueError(
            f"Voxel size for sample '{sample_id}' must be finite and greater than 0."
        )
    return voxel_size_mm


def require_voxel_sizes(
    voxel_sizes: Mapping[str, object],
    sample_ids: Iterable[object],
) -> dict[str, float]:
    """Validate and return the voxel-size map for exactly the requested samples."""
    normalized = {str(key): value for key, value in dict(voxel_sizes or {}).items()}
    required_ids = list(dict.fromkeys(str(sample_id) for sample_id in sample_ids))
    canonical_input: dict[str, list[str]] = {}
    for configured_id in normalized:
        canonical_id = configured_id.casefold()
        if canonical_id.startswith("total"):
            canonical_id = canonical_id[5:]
        canonical_input.setdefault(canonical_id, []).append(configured_id)

    resolved = {}
    missing = []
    for sample_id in required_ids:
        if sample_id in normalized:
            resolved[sample_id] = normalized[sample_id]
            continue
        canonical_id = sample_id.casefold()
        if canonical_id.startswith("total"):
            canonical_id = canonical_id[5:]
        matches = canonical_input.get(canonical_id, [])
        if len(matches) == 1:
            resolved[sample_id] = normalized[matches[0]]
        elif len(matches) > 1:
            raise ValueError(
                f"Ambiguous voxel-size entries for sample '{sample_id}': "
                + ", ".join(matches)
            )
        else:
            missing.append(sample_id)
    if missing:
        raise ValueError(
            "Missing measured voxel sizes (mm/voxel) for samples: "
            + ", ".join(missing)
        )
    return {
        sample_id: parse_voxel_size_mm(resolved[sample_id], sample_id)
        for sample_id in required_ids
    }
