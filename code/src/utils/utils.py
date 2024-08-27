import os
import errno

def ensure_path(path: str) -> bool:
    """Makes sure that the path is valid, creating it if necessary

    Args:
        path: the path to check

    Returns:
        bool: True on success, False if it fails
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            return True
        else:
            return False
    else:
        return True