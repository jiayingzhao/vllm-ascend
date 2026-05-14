# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vllm_ascend import envs


@dataclass(frozen=True)
class HIFP4FakeQuantConfig:
    group_size: int = 64
    qdim: int = -1


HIFP4_E1_8_COUNT = 8
HIFP4_E1_16_ELEMENTS = 4
HIFP4_E1_8_E1_16_COUNT = 2
HIFP4_MAX_LOCAL_VALUE = 7

E6M2_MIN_EXP = -48
E6M2_MAX_EXP = 15
E6M2_MAX_MANTISSA_CODE = 2
E6M2_MANTISSA_LEVELS = 4

S1P2_FRAC_LEVELS = 4
S1P2_MAX_VALUE = 1.75


def _round_half_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def _quantize_e6m2(scale: torch.Tensor) -> torch.Tensor:
    scale = torch.clamp(scale, min=2**E6M2_MIN_EXP)
    exponent = torch.floor(torch.log2(scale))
    mantissa = scale / torch.exp2(exponent) - 1
    mantissa_code = torch.floor(mantissa * E6M2_MANTISSA_LEVELS + 0.5)

    carry = mantissa_code >= E6M2_MANTISSA_LEVELS
    exponent = exponent + carry.to(exponent.dtype)
    mantissa_code = torch.where(carry, torch.zeros_like(mantissa_code), mantissa_code)

    overflow = (exponent > E6M2_MAX_EXP) | (
        (exponent == E6M2_MAX_EXP) & (mantissa_code > E6M2_MAX_MANTISSA_CODE)
    )
    exponent = torch.clamp(exponent, E6M2_MIN_EXP, E6M2_MAX_EXP)
    mantissa_code = torch.clamp(mantissa_code, 0, E6M2_MANTISSA_LEVELS - 1)
    mantissa_code = torch.where(
        (exponent == E6M2_MAX_EXP) & (mantissa_code > E6M2_MAX_MANTISSA_CODE),
        torch.full_like(mantissa_code, E6M2_MAX_MANTISSA_CODE),
        mantissa_code,
    )
    e6m2 = torch.exp2(exponent) * (1 + mantissa_code / E6M2_MANTISSA_LEVELS)
    return torch.where(overflow, torch.full_like(e6m2, float("nan")), e6m2)


def _quantize_s1p2(x: torch.Tensor) -> torch.Tensor:
    x = _round_half_away_from_zero(x * S1P2_FRAC_LEVELS) / S1P2_FRAC_LEVELS
    return torch.clamp(x, min=-S1P2_MAX_VALUE, max=S1P2_MAX_VALUE)


def fake_quant_hifp4_activation(
    x: torch.Tensor,
    config: HIFP4FakeQuantConfig = HIFP4FakeQuantConfig(),
) -> torch.Tensor:
    orig_dtype = x.dtype
    axis = config.qdim + x.ndim if config.qdim < 0 else config.qdim
    x = x.to(torch.float32).movedim(axis, -1)
    orig_shape = x.shape

    pad_size = (-orig_shape[-1]) % config.group_size
    if pad_size:
        x = F.pad(x, (0, pad_size), mode="constant", value=0)

    padded_shape = x.shape
    x = x.view(
        *padded_shape[:-1],
        -1,
        HIFP4_E1_8_COUNT,
        HIFP4_E1_8_E1_16_COUNT,
        HIFP4_E1_16_ELEMENTS,
    )
    abs_x = torch.abs(x)

    local_peak_16, _ = torch.max(abs_x, dim=-1, keepdim=True)
    local_peak_8, _ = torch.max(local_peak_16, dim=-2, keepdim=True)
    global_peak, _ = torch.max(local_peak_8, dim=-3, keepdim=True)

    scale = _quantize_e6m2(global_peak / HIFP4_MAX_LOCAL_VALUE)
    scale_recip = torch.reciprocal(scale)
    e1_8 = (local_peak_8 * scale_recip >= 4).to(x.dtype)
    e1_16 = (local_peak_16 * scale_recip * torch.exp2(-e1_8) >= 2).to(x.dtype)

    scaled = x * scale_recip * torch.exp2(-e1_8) * torch.exp2(-e1_16)
    s1p2 = _quantize_s1p2(scaled)
    out = s1p2 * scale * torch.exp2(e1_8) * torch.exp2(e1_16)
    out = out.flatten(-4, -1)

    if pad_size:
        out = out[..., : orig_shape[-1]]

    return out.reshape(orig_shape).movedim(-1, axis).to(orig_dtype)


def _prefix_matches_hifp4_activation(prefix: str) -> bool:
    patterns = envs.VLLM_ASCEND_HIFP4_ACTIVATION_FAKE_QUANT_PREFIXES
    for pattern in (item.strip() for item in patterns.split(",")):
        if not pattern:
            continue
        if prefix == pattern or prefix.endswith(f".{pattern}") or prefix.endswith(pattern):
            return True
    return False


def maybe_fake_quant_hifp4_activation(x: torch.Tensor, prefix: str) -> torch.Tensor:
    if not envs.VLLM_ASCEND_ENABLE_HIFP4_ACTIVATION_FAKE_QUANT:
        return x
    if not _prefix_matches_hifp4_activation(prefix):
        return x
    return fake_quant_hifp4_activation(x)
