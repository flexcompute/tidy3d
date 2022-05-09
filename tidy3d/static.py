from .log import Tidy3dError, log
from .components.base import Tidy3dBaseModel

class StaticException(Tidy3dError):
    """An exception when the tidy3d object has changed."""

def freeze_component(tidy3d_component) -> int:
    """Hashes an object if it's a tidy3dBaseModel.  Returns object and the hash value."""

    if not isinstance(tidy3d_component, Tidy3dBaseModel):
        return None

    object_hash = hash(tidy3d_component)
    tidy3d_component._hash = object_hash

    return object_hash

def unfreeze_component(tidy3d_component, object_hash) -> None:

    if not isinstance(tidy3d_component, Tidy3dBaseModel):
        return
    
    final_hash = hash(tidy3d_component)
    tidy3d_component._hash = None

    log.info(f'checking whether object has been modified')
    if final_hash != object_hash:
        raise StaticException(f"object has changed in static context.")


class MakeStatic:
    """Context manager that stores a hash and checks if the object has changed upon teardown."""

    def __init__(self, tidy3d_component):
        self.tidy3d_component = tidy3d_component

    def __enter__(self):
        self.original_hash = freeze_component(self.tidy3d_component)
        log.info(f'-> entering static context')
        return self.tidy3d_component

    def __exit__(self, *args):
        log.info(f'<- done with static context')
        unfreeze_component(self.tidy3d_component, self.original_hash)

def make_static(method):
    """Decorates a method to make any tidy3d objects static during the method call."""

    # stores the hashes of each of the args / kwargs
    hashes = {}

    def store_hash(obj):
        hash_value = freeze_component(obj)
        if hash_value is not None:
            hashes[obj] = hash_value        

    def method_static(*args, **kwargs):
        """The method in the static context."""

        # store the args and kwargs if they are tidy3d base model objects
        for arg in args:
            store_hash(arg)

        for value in kwargs.values():
            store_hash(value)

        # call original method
        return_value = method(*args, **kwargs)

        # unfreeze anything previously frozen
        for obj, hash_value in hashes.items():
            unfreeze_component(obj, hash_value)

        return return_value

    return method_static

