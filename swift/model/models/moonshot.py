# Copyright (c) ModelScope Contributors. All rights reserved.
from transformers import PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from whetstone.models import (KimiK25Config, KimiK25ForConditionalGeneration,
                              KimiK25PreTrainedModel, KimiK25Processor, KimiK25VisionProcessor,
                              load_kimi_tokenizer)

from swift.template import TemplateType
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..patcher import patch_get_input_embeddings
from ..register import ModelLoader, register_model


class KimiVLLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        KimiVLPreTrainedModel = get_class_from_dynamic_module('modeling_kimi_vl.KimiVLPreTrainedModel', model_dir)
        try:
            del KimiVLPreTrainedModel._supports_sdpa
        except AttributeError:
            pass
        model = super().get_model(model_dir, *args, **kwargs)
        patch_get_input_embeddings(model.vision_tower, 'patch_embed')
        return model


register_model(
    ModelMeta(
        MLLMModelType.kimi_vl,
        [
            ModelGroup([
                Model('moonshotai/Kimi-VL-A3B-Instruct', 'moonshotai/Kimi-VL-A3B-Instruct'),
                Model('moonshotai/Kimi-VL-A3B-Thinking', 'moonshotai/Kimi-VL-A3B-Thinking'),
                Model('moonshotai/Kimi-VL-A3B-Thinking-2506', 'moonshotai/Kimi-VL-A3B-Thinking-2506'),
            ])
        ],
        KimiVLLoader,
        template=TemplateType.kimi_vl,
        model_arch=ModelArch.llava_hf_legacy,
        architectures=['KimiVLForConditionalGeneration'],
        requires=['transformers<4.49'],
    ))


class KimiK25Loader(ModelLoader):

    def get_config(self, model_dir: str):
        self.auto_config_cls = KimiK25Config
        return super().get_config(model_dir)

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        try:
            del KimiK25PreTrainedModel._supports_sdpa
        except AttributeError:
            pass
        self.auto_model_cls = KimiK25ForConditionalGeneration
        model = super().get_model(model_dir, *args, **kwargs)
        patch_get_input_embeddings(model.vision_tower, 'patch_embed')
        return model

    def get_processor(self, model_dir: str, config):
        image_processor = KimiK25VisionProcessor.from_pretrained(model_dir, trust_remote_code=False)
        tokenizer = load_kimi_tokenizer(model_dir)
        chat_template = getattr(tokenizer, "chat_template", None)
        processor = KimiK25Processor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        processor.model_info = self.model_info
        processor.model_meta = self.model_meta
        return processor


MLLMModelType.kimi_k25 = 'kimi_k25'

register_model(
    ModelMeta(
        MLLMModelType.kimi_k25,
        [
            ModelGroup([
                Model('moonshotai/Kimi-K2.5', 'moonshotai/Kimi-K2.5'),
                Model('baseten/kimi-bf16-vision', 'baseten/kimi-bf16-vision'),
            ])
        ],
        KimiK25Loader,
        template=TemplateType.kimi_k25_native_sft,
        model_arch=ModelArch.llava_hf_legacy,
        architectures=['KimiK25ForConditionalGeneration'],
    ))
