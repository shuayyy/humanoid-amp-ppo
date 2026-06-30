_BLACKLIST_PKGS = ["utils", ".mdp"]

_TASKS_REGISTERED = False
_TASKS_REGISTERING = False


def register_tasks() -> None:
  global _TASKS_REGISTERED, _TASKS_REGISTERING

  if _TASKS_REGISTERED or _TASKS_REGISTERING:
    return

  _TASKS_REGISTERING = True
  try:
    from mjlab.utils.lab_api.tasks.importer import import_packages

    import_packages(__name__, _BLACKLIST_PKGS)
    _TASKS_REGISTERED = True
  finally:
    _TASKS_REGISTERING = False
