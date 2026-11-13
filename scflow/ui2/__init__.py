def launch(*args, **kwargs):
    from .main_window import launch as _launch
    return _launch(*args, **kwargs)

__all__ = ["launch"]
