from typing import Union

import torch

def assert_same_shape(expected, value):
    assert torch.is_tensor(value), "value should be a tensor"
    assert expected.shape == value.shape, "expected shape is " + expected.shape


def real_to_range(val: torch.Tensor, r_min: float, r_max: float):
    d = r_max - r_min
    return torch.abs(torch.remainder(val - r_min - d, d * 2) - d) + r_min


RectangleDimentions =  Union[int, tuple[int, int]]

def as_img_dims(t: RectangleDimentions) -> tuple[int, int]:
    if isinstance(t, tuple):
        return t
    return (t,) * 2
