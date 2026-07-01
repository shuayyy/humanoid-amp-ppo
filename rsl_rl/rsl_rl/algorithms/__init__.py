# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning algorithms."""

from .amp_ppo import AMP_PPO
from .ppo import PPO

__all__ = ["PPO", "AMP_PPO"]
