# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.template import TemplateType

from ..constant import LLMModelType
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model


register_model(
    ModelMeta(
        LLMModelType.nemotron_h,
        [
            ModelGroup([
                Model(
                    None,
                    'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16',
                ),
            ]),
        ],
        ModelLoader,
        template=TemplateType.default,
        architectures=['NemotronHForCausalLM'],
        model_arch=None,
        requires=['transformers>=4.57.0', 'mamba-ssm', 'causal-conv1d'],
    ))
