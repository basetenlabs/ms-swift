"""
Kimi-K2.5 native SFT template.

Bypasses ms-swift's standard template pipeline (format strings, agent templates,
tool-call rewriting) and instead uses the official Kimi-K2.5 chat template
directly via tokenizer.apply_chat_template().

This produces token sequences identical to what the model was pretrained on,
including native tool-call tokens (<|tool_calls_section_begin|>, etc.),
TypeScript-format tool declarations, and <think></think> tags.

Loss masking is built by scanning for assistant-body spans in the token
sequence, not by regex or ReAct pattern matching.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import Template
from ..constant import LLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Prompt

# Import pure-Python helpers from whetstone (no torch/swift deps).
# These are shared between the template and local tests.
from whetstone.kimi_sft_utils import (
    ASSISTANT_PREFIX_IDS,
    IM_END_ID,
    build_labels,
    build_labels_for_ordinals,
    compute_trainable_ordinals,
    find_assistant_body_spans,
    verify_token_ids,
)


class KimiK25NativeSFTTemplate(Template):
    support_padding_free = True

    """Template that uses the official Kimi-K2.5 chat template for SFT.

    Overrides _encode_truncated() to bypass all ms-swift preprocessing
    (_preprocess_inputs, _preprocess_function_call, _swift_prepare_inputs)
    and instead:
    1. Renders with tokenizer.apply_chat_template()
    2. Tokenizes the rendered text
    3. Builds labels by scanning for assistant body spans
    4. Handles truncation
    """

    _token_ids_verified = False

    def _verify_token_ids(self):
        """One-time check that hardcoded token IDs match this tokenizer."""
        if KimiK25NativeSFTTemplate._token_ids_verified:
            return
        verify_token_ids(self.tokenizer)
        KimiK25NativeSFTTemplate._token_ids_verified = True

    @staticmethod
    def _sanitize_messages(messages: list) -> list:
        """Fix HF datasets artifacts before rendering.

        HF Arrow schema normalization can:
        - Set reasoning_content=None (should be absent or '')
        - Add image_url=null to text content parts
        These cause incorrect rendering with the Kimi Jinja template.
        """
        sanitized = []
        for msg in messages:
            msg = dict(msg)
            # reasoning_content=None → Jinja renders <think>None</think>
            if msg.get('reasoning_content') is None and 'reasoning_content' in msg:
                del msg['reasoning_content']
            # Clean null fields from content parts
            content = msg.get('content')
            if isinstance(content, list):
                msg['content'] = [
                    {k: v for k, v in part.items() if v is not None}
                    if isinstance(part, dict) else part
                    for part in content
                ]
            sanitized.append(msg)
        return sanitized

    def _render_and_tokenize(self, messages: list, tools: list | None) -> List[int]:
        """Render messages with official Kimi chat template, then tokenize."""
        import json as _json
        messages = self._sanitize_messages(messages)
        kwargs = {}
        if tools:
            # HF datasets may store tools as a JSON string; deserialize if needed
            if isinstance(tools, str):
                tools = _json.loads(tools)
            kwargs['tools'] = tools
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            thinking=True,
            **kwargs,
        )
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return input_ids

    def _build_labels(self, input_ids: List[int]) -> List[int]:
        """Build labels: train on assistant body spans, mask everything else."""
        return build_labels(input_ids)

    def _encode_truncated(self, inputs: StdTemplateInputs):
        """Full encoding path: raw messages → {input_ids, labels, loss_scale}.

        Bypasses _preprocess_inputs() entirely so that:
        - tool_calls on assistant messages are preserved (not converted to ReAct)
        - tool messages are not merged or reformatted
        - system prompt is not replaced with agent template instructions

        Uses train_mode (default "suffix") and prefix_length to decide which
        assistant messages to train on:
        - "suffix": requires prefix_length; trains assistant messages at
          index >= prefix_length
        - "last": trains only the final assistant message; prefix_length
          is ignored if absent
        """
        self._verify_token_ids()

        # Determine train_mode.  Priority:
        #   1. Config-level self.train_mode (from YAML / get_template())
        #   2. Per-row field (train_mode in dataset row → extra_kwargs)
        #   3. "suffix" if prefix_length is present, "last" otherwise
        prefix_length = inputs.extra_kwargs.get('prefix_length')
        train_mode = (
            self.train_mode
            or inputs.extra_kwargs.get('train_mode')
        )
        if train_mode is None:
            train_mode = 'suffix' if prefix_length is not None else 'last'
        if prefix_length is None and train_mode == 'suffix':
            raise ValueError(
                "kimi_k25_native_sft template requires 'prefix_length' on every example "
                "when train_mode='suffix'. Set it in the dataset row as a top-level field, "
                "or use train_mode='last'."
            )

        # Compute which assistant ordinals are trainable.
        # StdTemplateInputs.from_dict() strips a leading system message out of
        # inputs.messages but leaves extra_kwargs untouched, so dataset
        # prefix_length still counts that system slot when present.
        raw_messages = inputs.messages
        if prefix_length is not None:
            raw_prefix_length = prefix_length - (1 if inputs.system is not None else 0)
        else:
            raw_prefix_length = 0
        trainable_ordinals = compute_trainable_ordinals(
            raw_messages, raw_prefix_length, train_mode=train_mode,
        )
        n_assistant = sum(1 for m in raw_messages if m['role'] == 'assistant')

        # Reconstruct full message list with system message for rendering
        messages = deepcopy(raw_messages)
        if inputs.system is not None:
            messages.insert(0, {'role': 'system', 'content': inputs.system})

        # Render and tokenize using official Kimi template
        input_ids = self._render_and_tokenize(messages, inputs.tools)

        # Build labels: only suffix assistant spans are supervised
        labels = build_labels_for_ordinals(input_ids, trainable_ordinals, n_assistant)

        # Mask first token (ms-swift convention: first token never contributes to loss)
        if labels and labels[0] != -100:
            labels[0] = -100

        # Handle truncation
        loss_scale = None
        length = len(input_ids)

        if self.max_length is not None and length > self.max_length:
            if self.truncation_strategy in {'right', 'left'}:
                input_ids, labels, loss_scale = self._truncate(
                    input_ids, labels, loss_scale,
                    truncation_strategy=self.truncation_strategy,
                )
                length = len(input_ids)
            elif self.truncation_strategy == 'raise':
                from ..base import MaxLengthError
                raise MaxLengthError(
                    f'Current length of row({length}) is larger'
                    f' than the max_length({self.max_length}).'
                )
            elif self.truncation_strategy == 'split':
                batched = []
                i = 0
                while i < length:
                    chunk_ids = input_ids[i:i + self.max_length]
                    chunk_labels = labels[i:i + self.max_length]
                    if chunk_labels and len(chunk_labels) > 0:
                        chunk_labels[0] = -100
                    chunk_len = len(chunk_ids)
                    batched.append({
                        'input_ids': chunk_ids,
                        'labels': chunk_labels,
                        'loss_scale': None,
                        'length': chunk_len,
                    })
                    i += self.max_length
                return batched
            else:
                raise ValueError(
                    f'Invalid truncation_strategy: {self.truncation_strategy}'
                )

        return {
            'input_ids': input_ids,
            'labels': labels,
            'loss_scale': loss_scale,
            'length': length,
        }


# -- Template metadata and registration --

@dataclass
class KimiK25NativeSFTMeta(TemplateMeta):
    """Metadata for the native Kimi-K2.5 SFT template.

    The format strings here are placeholders — they are never used because
    KimiK25NativeSFTTemplate overrides _encode_truncated() and bypasses
    the swift template rendering entirely. They exist only to satisfy
    TemplateMeta's structural requirements.
    """
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ['{{QUERY}}'])
    chat_sep: Optional[Prompt] = field(default_factory=list)
    suffix: Prompt = field(default_factory=list)
    default_system: Optional[str] = None


# Register under a distinct name so it doesn't conflict with stock kimi_k2
LLMTemplateType.kimi_k25_native_sft = 'kimi_k25_native_sft'

register_template(KimiK25NativeSFTMeta(
    LLMTemplateType.kimi_k25_native_sft,
    template_cls=KimiK25NativeSFTTemplate,
))
