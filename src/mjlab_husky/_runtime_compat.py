"""Compatibility shims for upstream dependency API drift.

These patches are intentionally narrow and only fill symbols that older/newer
`mjlab` and `mujoco_warp` builds expect during import/runtime startup.
"""

from __future__ import annotations

import types

import mujoco
import warp as wp


def _patch_mujoco_enable_bits() -> None:
  # Some `mujoco_warp` builds reference this enum member during import even
  # when the bundled MuJoCo wheel does not expose it.
  if not hasattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD"):
    setattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD", 0)


def _patch_warp_context() -> None:
  # Newer `warp-lang` exposes `get_cuda_driver_version()` instead of the older
  # `warp.context.runtime.driver_version` path used by `mjlab`.
  if hasattr(wp, "context"):
    return

  driver_version = None
  if hasattr(wp, "get_cuda_driver_version"):
    try:
      driver_version = wp.get_cuda_driver_version()
    except Exception:
      driver_version = None

  wp.context = types.SimpleNamespace(  # type: ignore[attr-defined]
    runtime=types.SimpleNamespace(driver_version=driver_version)
  )


def apply_runtime_compat() -> None:
  _patch_mujoco_enable_bits()
  _patch_warp_context()
