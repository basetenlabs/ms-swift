# Copyright (c) ModelScope Contributors. All rights reserved.
import math
from megatron.core.models.mamba import MambaModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from typing import Optional

from swift.model import ModelType
from swift.utils import get_logger

from ..constant import MegatronModelType
from ..gpt_bridge import GPTBridge
from ..register import MegatronModelLoader, MegatronModelMeta, register_megatron_model

logger = get_logger()


class NemotronHBridge(GPTBridge):
    """Minimal Nemotron-H bridge.

    This initial spike is intended for the pre-converted Megatron checkpoint
    path (`--mcore_model ...`). Full HF<->Megatron conversion for Nemotron-H
    requires the custom Mamba/MoE/MTP mappings from NVIDIA Megatron-Bridge and
    should be added as a separate follow-up.  The target Tailor Swift flow uses
    `merge_lora=true` for the final artifact, but merged safetensors export
    still needs a GPU smoke test against the real Nemotron-H checkpoint.
    """

    hf_layers_prefix = 'backbone.layers'
    hf_mtp_prefix = 'mtp.layers'
    hf_embed_key = 'backbone.embeddings.weight'
    hf_final_layernorm_key = 'backbone.norm_f.weight'
    hf_lm_head_key = 'lm_head.weight'


class NemotronHLoader(MegatronModelLoader):
    """Build NVIDIA Nemotron-H as an MCore MambaModel.

    ms-swift's default MegatronModelLoader builds GPTModel stacks. Nemotron 3
    Super instead uses Megatron-Core's hybrid MambaModel with a layer pattern
    containing Mamba (M), attention (*), and MoE (E) blocks, plus a repeated MTP
    block. The config defaults are set in `convert_hf_config`; this loader wires
    them into MambaModel.
    """

    @staticmethod
    def _build_hybrid_layer_pattern(config) -> Optional[str]:
        pattern = getattr(config, 'hybrid_override_pattern', None)
        if not pattern:
            return None

        sep = Symbols.MTP_SEPARATOR
        main_pattern = pattern.split(sep)[0]
        mtp_pattern = getattr(config, 'mtp_hybrid_override_pattern', None)
        mtp_num_layers = getattr(config, 'mtp_num_layers', None) or 0
        mtp_use_repeated_layer = getattr(config, 'mtp_use_repeated_layer', False)

        if mtp_pattern:
            if mtp_use_repeated_layer:
                num_pattern_copies = max(1, mtp_num_layers)
            else:
                num_pattern_copies = mtp_num_layers
            if num_pattern_copies:
                return main_pattern + sep + sep.join([mtp_pattern] * num_pattern_copies)
        return main_pattern

    def build_model(
        self,
        pre_process=True,
        post_process=True,
        vp_stage: Optional[int] = None,
    ) -> MambaModel:
        config = self.config

        if getattr(config, 'virtual_pipeline_model_parallel_size', None) is not None or vp_stage is not None:
            raise ValueError('Virtual pipeline model parallelism is not supported for Nemotron-H/MambaModel yet.')

        vocab_size = math.ceil(
            config.padded_vocab_size / config.tensor_model_parallel_size) * config.tensor_model_parallel_size
        max_sequence_length = config.max_position_embeddings or getattr(self.args, 'max_length', None) or 8192
        hybrid_layer_pattern = self._build_hybrid_layer_pattern(config)
        logger.info(f'Nemotron-H hybrid_layer_pattern: {hybrid_layer_pattern}')

        model = MambaModel(
            config=config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            hybrid_override_pattern=hybrid_layer_pattern,
            fp16_lm_cross_entropy=False,
            parallel_output=True,
            share_embeddings_and_output_weights=not config.untie_embeddings_and_output_weights,
            position_embedding_type=config.position_embedding_type,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rotary_base,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        model.args = self.args
        return model


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.nemotron_h,
        [ModelType.nemotron_h],
        bridge_cls=NemotronHBridge,
        loader=NemotronHLoader,
    ))
