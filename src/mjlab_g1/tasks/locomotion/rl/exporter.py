import os

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab_g1.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)


def export_locomotion_policy_as_onnx(
  actor: object,
  path: str,
  normalizer: object | None = None,
  filename="policy.onnx",
  verbose=False,
):
  del normalizer
  onnx_model = actor.as_onnx(verbose=verbose)
  onnx_model.to("cpu")
  onnx_model.eval()

  os.makedirs(path, exist_ok=True)
  save_path = os.path.join(path, filename)

  torch.onnx.export(
    onnx_model,
    onnx_model.get_dummy_inputs(),
    save_path,
    export_params=True,
    opset_version=18,
    verbose=verbose,
    input_names=onnx_model.input_names,
    output_names=onnx_model.output_names,
  )


def attach_onnx_metadata(
  env: ManagerBasedRlEnv, run_path: str, path: str, filename="policy.onnx"
) -> None:
  """Attach locomotion-specific metadata to ONNX model.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.
    path: Directory containing the ONNX file.
    filename: Name of the ONNX file.
  """
  onnx_path = os.path.join(path, filename)
  metadata = get_base_metadata(env, run_path)
  attach_metadata_to_onnx(onnx_path, metadata)
