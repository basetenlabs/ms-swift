# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from PIL import Image
from whetstone.models import KimiK25ForConditionalGeneration

from swift.model import ModelType
from ..constant import MegatronModelType
from ..gpt_bridge import MultimodalGPTBridge
from ..register import MegatronModelMeta, register_megatron_model
from .utils import HuggingFaceModule


class KimiK25Bridge(MultimodalGPTBridge):
    hf_layers_prefix = 'language_model.model.layers'
    hf_embed_key = 'language_model.model.embed_tokens.weight'
    hf_final_layernorm_key = 'language_model.model.norm.weight'
    hf_lm_head_key = 'language_model.lm_head.weight'
    hf_score_key = 'language_model.score.weight'


class KimiK25Vit(HuggingFaceModule):
    module_mapping = {'vision_tower': 'vision_tower', 'multi_modal_projector': 'multi_modal_projector'}
    _vision_tower = ['vision_tower']
    _aligner = ['multi_modal_projector']

    def __init__(self, config):
        super().__init__(config, [KimiK25ForConditionalGeneration])

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        model = self._hf_model[0]
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values')
        grid_thws = kwargs.get('grid_thws', kwargs.get('image_grid_hws'))
        if pixel_values is not None and pixel_values.size(0) > 0:
            pixel_values = pixel_values.to(model.vision_tower.dtype)
            if grid_thws is None:
                raise KeyError("Missing grid_thws/image_grid_hws for Kimi K2.5 vision inputs")
            image_features: torch.Tensor = model._extract_image_features(pixel_values, grid_thws)
            inputs_embeds = inputs_embeds.to(image_features[0].dtype).clone()
            inputs_embeds = model._merge_with_image_features(inputs_embeds, input_ids, image_features)
        else:
            image_processor = self.processor.image_processor
            dummy_image = Image.new('RGB', (32, 32), (0, 0, 0))
            image_inputs = image_processor([{'type': 'image', 'image': dummy_image}], return_tensors='pt')
            pixel_values = image_inputs['pixel_values'].to(device=inputs_embeds.device, dtype=model.vision_tower.dtype)
            grid_thws_dummy = image_inputs.get('grid_thws', image_inputs['image_grid_hws']).to(inputs_embeds.device)
            image_features: torch.Tensor = model._extract_image_features(pixel_values, grid_thws_dummy)
            inputs_embeds = inputs_embeds + image_features.mean() * 0.
        return inputs_embeds


MegatronModelType.kimi_k25 = 'kimi_k25'

register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.kimi_k25,
        [
            'kimi_k25',
        ],
        bridge_cls=KimiK25Bridge,
        visual_cls=KimiK25Vit,
    ))
