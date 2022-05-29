# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generic utilities
"""

import inspect
from collections import OrderedDict, UserDict
from contextlib import ExitStack
from dataclasses import fields
from enum import Enum
from typing import Any, ContextManager, List, Tuple

import numpy as np

from .import_utils import is_flax_available, is_tf_available, is_torch_available, is_torch_fx_proxy


class cached_property(property):
    """
    Descriptor that mimics @property but caches output in member variable.

    From tensorflow_datasets

    Built-in in functools from Python 3.8.
    """

    def __get__(self, obj, objtype=None):
        # See docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        # 优先从 obj 的属性中获取, 如果获取不到, 则调用 fget 获取, 并缓存在 obj 的属性中
        cached = getattr(obj, attr, None)
        if cached is None:
            cached = self.fget(obj)
            setattr(obj, attr, cached)
        return cached


def is_tensor(x):
    """
    Tests if `x` is a `torch.Tensor`, `tf.Tensor`, `jaxlib.xla_extension.DeviceArray` or `np.ndarray`.
    """
    if is_torch_fx_proxy(x):
        return True
    if is_torch_available():
        import torch

        if isinstance(x, torch.Tensor):
            return True
    if is_tf_available():
        import tensorflow as tf

        if isinstance(x, tf.Tensor):
            return True

    if is_flax_available():
        import jax.numpy as jnp
        from jax.core import Tracer

        if isinstance(x, (jnp.ndarray, Tracer)):
            return True

    return isinstance(x, np.ndarray)


def _is_numpy(x):
    return isinstance(x, np.ndarray)


def _is_torch(x):
    import torch

    return isinstance(x, torch.Tensor)


def _is_torch_device(x):
    import torch

    return isinstance(x, torch.device)


def _is_tensorflow(x):
    import tensorflow as tf

    return isinstance(x, tf.Tensor)


def _is_jax(x):
    import jax.numpy as jnp  # noqa: F811

    return isinstance(x, jnp.ndarray)


def to_py_obj(obj):
    """
    转换成 python 类型
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a python list.
    """
    # 对于 dict 和 list 使用递归调用
    if isinstance(obj, (dict, UserDict)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_py_obj(o) for o in obj]
    # 对于四种框架的类型进行转换, 基本都可以说是先转成 numpy 格式, 然后使用 tolist 方法
    elif is_tf_available() and _is_tensorflow(obj):
        return obj.numpy().tolist()
    elif is_torch_available() and _is_torch(obj):
        return obj.detach().cpu().tolist()
    elif is_flax_available() and _is_jax(obj):
        return np.asarray(obj).tolist()
    elif isinstance(obj, (np.ndarray, np.number)):  # tolist also works on 0d np arrays
        return obj.tolist()
    else:
        return obj


def to_numpy(obj):
    """
    转换成 numpy 类型
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a Numpy array.
    """
    if isinstance(obj, (dict, UserDict)):
        return {k: to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return np.array(obj)
    elif is_tf_available() and _is_tensorflow(obj):
        return obj.numpy()
    elif is_torch_available() and _is_torch(obj):
        return obj.detach().cpu().numpy()
    elif is_flax_available() and _is_jax(obj):
        return np.asarray(obj)
    else:
        return obj


class ModelOutput(OrderedDict):
    """
    定义模型输出, 继承自有序字典
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __post_init__(self):
        # 获取所有的字段
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        # 除了第一个字段外, 都需要有默认值
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        # 如果其他字段都是 None
        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                # 尝试迭代
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    # 每次迭代必须要有两个字段, 且第一个字段的类型是 str
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    # 更新属性
                    setattr(self, element[0], element[1])
                    # 如果不是 None, 就作为字典存储进去
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                # first_field 不可迭代, 且不是 None, 就直接存储进字典中
                self[class_fields[0].name] = first_field
        else:
            # 如果不是 None 的字段, 就作为字典的 key value 存储进去
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            # 这个就是支持索引位置的
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        # 字典中只保存非 None 的值
        if name in self.keys() and value is not None:
            # 要防止递归, 所以只能用 super() 的 __setitem__ 方法
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        # 属性可以直接设置
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        # 转换成 tuple, 使用的 keys 就表示是从字典中获取, 字典中又是不存 None 值的, 所以只返回有值的属性
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the `padding` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in an
    IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"


class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(self, context_managers: List[ContextManager]):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)


def find_labels(model_class):
    """
    Find the labels used by a given model.

    Args:
        model_class (`type`): The class of the model.
    """
    model_name = model_class.__name__
    if model_name.startswith("TF"):
        signature = inspect.signature(model_class.call)
    elif model_name.startswith("Flax"):
        signature = inspect.signature(model_class.__call__)
    else:
        signature = inspect.signature(model_class.forward)
    if "QuestionAnswering" in model_name:
        return [p for p in signature.parameters if "label" in p or p in ("start_positions", "end_positions")]
    else:
        return [p for p in signature.parameters if "label" in p]
