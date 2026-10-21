"""Product models in ``tidy3d.web``.

A product package submits its own models through ``tidy3d.web`` by listing their classes and
task types in ``task_types`` on its
:class:`~tidy3d.web.core.client_identity.ClientIdentity`, and reads its own classes for the
type tags it shares with Tidy3D by naming them in ``model_classes``. The component classes carry
no task-type knowledge. Nothing here is imported by ``tidy3d.components``: ``Step.operation``
keeps the built-in union, and :class:`~tidy3d.web.api.container.Job` builds the single step of
a product model itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Any, TypeVar, get_args

from pydantic import SerializeAsAny, WrapValidator

from tidy3d.components.base import TYPE_TO_CLASS_MAP, Tidy3dBaseModel
from tidy3d.components.types.base import TYPE_TAG_STR, discriminated_union
from tidy3d.components.types.workflow import WorkflowOperationType
from tidy3d.components.workflow import Step, Workflow, resolve_workflow
from tidy3d.web.core.client_identity import get_client_identity

if TYPE_CHECKING:
    from os import PathLike

    from pydantic import ValidatorFunctionWrapHandler

_Entry = TypeVar("_Entry")


def most_specific_entry(table: Mapping[type, _Entry], cls: type) -> _Entry | None:
    """Return the entry of the nearest class in ``cls``'s MRO that ``table`` lists, if any.

    A subclass listed in the table wins over its listed base, whatever the order of the table.
    """
    for base in cls.__mro__:
        if base in table:
            return table[base]
    return None


def product_task_type_of(cls: object) -> str | None:
    """Return the task type the active product client lists for ``cls``, if any."""
    identity = get_client_identity()
    if identity is None or not isinstance(cls, type):
        return None
    return most_specific_entry(identity.task_types, cls)


def is_workflow_model_class(cls: object) -> bool:
    """Return whether ``cls`` is a product model class of the active client."""
    return product_task_type_of(cls) is not None


def is_product_model(value: object) -> bool:
    """Return whether ``value`` is a product model of the active client."""
    return is_workflow_model_class(type(value))


def is_workflow_operation(value: object) -> bool:
    """Return whether ``value`` is a built-in workflow operation or a product model."""
    return isinstance(value, WorkflowOperationType) or is_product_model(value)


def is_foreign_model(value: object) -> bool:
    """Return whether ``value`` is a model Tidy3D cannot submit as its own.

    That is a model outside the built-in operations, or a product twin: a subclass of one
    whose type tag Tidy3D reads back as another class.
    """
    if not isinstance(value, WorkflowOperationType):
        return True
    cls = type(value)
    # Every model is tagged with its class name.
    return TYPE_TO_CLASS_MAP.get(cls.__name__) is not cls


@lru_cache(maxsize=1)
def _built_in_operation_tags() -> frozenset[str]:
    """Return the type tags ``Step.operation``'s built-in union accepts."""

    def flatten(type_hint: object) -> tuple[type, ...]:
        args = get_args(type_hint)
        if not args:
            return (type_hint,) if isinstance(type_hint, type) else ()
        return tuple(cls for arg in args for cls in flatten(arg))

    # Every model is tagged with its class name.
    return frozenset(cls.__name__ for cls in flatten(WorkflowOperationType))


def resolve_product_workflow(
    simulation: WorkflowOperationType, workflow: Workflow | None = None
) -> Workflow:
    """Resolve the workflow of a submission, building a product model's single step without validation.

    ``Step.operation`` is typed with the built-in union, so a product model whose type tag the
    union does not accept cannot pass the step's validation. That covers models outside the
    union and listed subclasses of Tidy3D models under their own tag. The web entry points have
    already accepted the model, and the step carries nothing else to validate. Product twins
    keep Tidy3D's tag and resolve as the built-in class does.
    """
    if (
        workflow is None
        and is_product_model(simulation)
        and type(simulation).__name__ not in _built_in_operation_tags()
    ):
        step = Step.model_construct(name="execute", operation=simulation, cacheable=True)
        return Workflow.model_construct(steps=(step,))
    return resolve_workflow(simulation, workflow)


def _product_model_classes() -> Mapping[str, type[Tidy3dBaseModel]]:
    """Return the classes the active product client reads for the tags it shares with Tidy3D."""
    identity = get_client_identity()
    return {} if identity is None else identity.model_classes


def product_model_class_of_file(path: PathLike) -> type[Tidy3dBaseModel] | None:
    """Return the product class that reads the model file at ``path``, if the active client names one."""
    classes = _product_model_classes()
    if not classes:
        return None
    type_tag = Tidy3dBaseModel.dict_from_file(path, load_data_arrays=False).get(TYPE_TAG_STR)
    return classes.get(type_tag) if isinstance(type_tag, str) else None


def serialized_model_class(value: object) -> type[Tidy3dBaseModel] | None:
    """Return the class a serialized model's type tag names for the active client, if any.

    A product client reads its own classes for the tags it shares with Tidy3D; every other
    tag resolves through Tidy3D's registry.
    """
    if not isinstance(value, Mapping) or not isinstance(value.get(TYPE_TAG_STR), str):
        return None
    type_tag = value[TYPE_TAG_STR]
    return _product_model_classes().get(type_tag) or TYPE_TO_CLASS_MAP.get(type_tag)


def validate_workflow_operation(value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
    """Pydantic wrap validator accepting workflow models besides the built-in operations.

    A product model instance is accepted as it is. A serialized model whose type tag names a
    product model class is built with that class; that is the product's class for a tag Tidy3D
    also defines. Everything else goes through the built-in union.
    """
    if is_product_model(value):
        return value
    model_cls = serialized_model_class(value)
    if model_cls is not None and is_workflow_model_class(model_cls):
        return model_cls.model_validate(value)
    return handler(value)


# Field annotation accepting the built-in workflow operations and workflow models. Values
# serialize as their own class, so fields the built-in union does not know survive a dump.
WorkflowOperationInput = SerializeAsAny[
    Annotated[
        discriminated_union(WorkflowOperationType), WrapValidator(validate_workflow_operation)
    ]
]
