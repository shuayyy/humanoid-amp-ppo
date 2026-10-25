"""SuperMimic (Liao et al., arXiv 2610.28480) learner for mjlab G1 motion tracking.

Kept import-free on purpose: submodules pull in torch / mujoco / mjlab only when imported
directly (e.g. `from mjlab_g1.rl.supermimic.features import goal`).
"""
