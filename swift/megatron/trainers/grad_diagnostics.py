"""Diagnostic patch for grad_norm=0 investigation.

Add to grpo_trainer.py imports:
    from .grad_diagnostics import patch_grad_diagnostics

Call in __init__ after super().__init__:
    patch_grad_diagnostics(self)
"""
import torch
from megatron.core import mpu
from swift.utils import get_logger

logger = get_logger()

_DIAG_STEPS = 5  # Only log diagnostics for the first N steps
_diag_step_counter = [0]
_grad_hooks = []


def patch_grad_diagnostics(trainer):
    """Monkey-patch trainer to add gradient flow diagnostics."""

    original_loss_func = trainer.__class__.loss_func
    original_train_step = trainer.__class__.train_step

    # Register gradient hooks on LoRA params to detect when autograd computes their gradients
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    
    if rank == 0:
        lora_param_count = 0
        trainable_param_count = 0
        has_main_grad_count = 0
        in_bucket_group = 0
        
        # Check wrapped model (DDP-wrapped)
        wrapped_model = trainer.wrapped_models[0]
        ddp_model = wrapped_model
        
        for name, param in trainer.unwrapped_models[0].named_parameters():
            if param.requires_grad:
                trainable_param_count += 1
                if hasattr(param, 'main_grad'):
                    has_main_grad_count += 1
                if 'lora' in name.lower():
                    lora_param_count += 1
        
        # Check if DDP has param_to_bucket_group
        if hasattr(ddp_model, 'param_to_bucket_group'):
            in_bucket_group = len(ddp_model.param_to_bucket_group)
        
        logger.info(f"[GRAD_DIAG] Setup: trainable_params={trainable_param_count}, "
                     f"lora_params={lora_param_count}, "
                     f"has_main_grad={has_main_grad_count}, "
                     f"in_bucket_group={in_bucket_group}")
        
        # Check a few specific LoRA params
        sample_count = 0
        for name, param in trainer.unwrapped_models[0].named_parameters():
            if 'lora' in name.lower() and param.requires_grad and sample_count < 3:
                logger.info(f"[GRAD_DIAG] LoRA param '{name}': "
                             f"shape={list(param.shape)}, dtype={param.dtype}, "
                             f"has_main_grad={hasattr(param, 'main_grad')}, "
                             f"in_bucket_group={param in ddp_model.param_to_bucket_group if hasattr(ddp_model, 'param_to_bucket_group') else 'N/A'}")
                sample_count += 1
        
        # Register hooks on first 2 LoRA params to detect gradient computation
        hooked = 0
        for name, param in trainer.unwrapped_models[0].named_parameters():
            if 'lora' in name.lower() and param.requires_grad and hooked < 2:
                param_name = name
                def make_hook(pname):
                    def hook(grad):
                        step = _diag_step_counter[0]
                        if step < _DIAG_STEPS:
                            logger.info(f"[GRAD_DIAG step={step}] GRAD COMPUTED for '{pname}': "
                                         f"grad.abs().max()={grad.abs().max().item():.8e}, "
                                         f"grad.abs().mean()={grad.abs().mean().item():.8e}")
                        return grad  # don't modify
                    return hook
                handle = param.register_hook(make_hook(param_name))
                _grad_hooks.append(handle)
                hooked += 1
                logger.info(f"[GRAD_DIAG] Registered grad hook on '{param_name}'")

    def diagnostic_loss_func(self, output_tensor, data=None, **kwargs):
        step = _diag_step_counter[0]
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if step < _DIAG_STEPS and rank == 0:
            per_token_logps = data.get('per_token_logps') if data else None
            if per_token_logps is not None:
                logger.info(f"[GRAD_DIAG step={step}] per_token_logps: "
                             f"requires_grad={per_token_logps.requires_grad}, "
                             f"grad_fn={type(per_token_logps.grad_fn).__name__ if per_token_logps.grad_fn else None}, "
                             f"shape={list(per_token_logps.shape)}, "
                             f"abs_max={per_token_logps.abs().max().item():.6f}")
                
                # Check advantages
                advantages = data.get('advantages')
                if advantages is not None:
                    logger.info(f"[GRAD_DIAG step={step}] advantages: "
                                 f"shape={list(advantages.shape)}, "
                                 f"nonzero_count={(advantages != 0).sum().item()}/{advantages.numel()}, "
                                 f"abs_max={advantages.abs().max().item():.6f}")
                
                # Check old_per_token_logps
                old_logps = data.get('old_per_token_logps')
                if old_logps is not None:
                    logger.info(f"[GRAD_DIAG step={step}] old_per_token_logps: "
                                 f"requires_grad={old_logps.requires_grad}, "
                                 f"abs_max={old_logps.abs().max().item():.6f}")
            else:
                logger.info(f"[GRAD_DIAG step={step}] per_token_logps is None!")

        result = original_loss_func(self, output_tensor, data=data, **kwargs)

        if step < _DIAG_STEPS and rank == 0:
            loss = result[0]
            logger.info(f"[GRAD_DIAG step={step}] GRPO loss: "
                         f"value={loss.item():.8f}, "
                         f"requires_grad={loss.requires_grad}, "
                         f"grad_fn={type(loss.grad_fn).__name__ if loss.grad_fn else None}")

        return result

    def diagnostic_train_step(self, train_data_iterator):
        step = _diag_step_counter[0]
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # Check main_grad BEFORE optimizer step (after forward_backward_func)
        if step < _DIAG_STEPS and rank == 0:
            # We need to hook into the forward_backward_func to check after backward
            # but before optimizer. Let's check just before optimizer.step in the original train_step.
            pass

        metrics, grad_norm = original_train_step(self, train_data_iterator)

        if step < _DIAG_STEPS and rank == 0:
            logger.info(f"[GRAD_DIAG step={step}] ===== POST-OPTIMIZER DIAGNOSTICS =====")
            logger.info(f"[GRAD_DIAG step={step}] grad_norm={grad_norm}")

            # Check main_grad and param.grad on LoRA params
            lora_params_total = 0
            lora_main_grad_nonzero = 0
            lora_grad_nonzero = 0
            lora_main_grad_exists = 0

            for name, param in self.unwrapped_models[0].named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    lora_params_total += 1
                    if hasattr(param, 'main_grad'):
                        lora_main_grad_exists += 1
                        if param.main_grad.abs().max() > 0:
                            lora_main_grad_nonzero += 1
                    if param.grad is not None and param.grad.abs().max() > 0:
                        lora_grad_nonzero += 1
                    
                    # Log first few LoRA params in detail
                    if lora_params_total <= 3:
                        mg_max = param.main_grad.abs().max().item() if hasattr(param, 'main_grad') else -1
                        pg_max = param.grad.abs().max().item() if param.grad is not None else -1
                        logger.info(f"[GRAD_DIAG step={step}] LoRA '{name}': "
                                     f"main_grad_max={mg_max:.8e}, "
                                     f"param.grad_max={pg_max:.8e}")

            logger.info(f"[GRAD_DIAG step={step}] Summary: "
                         f"lora_total={lora_params_total}, "
                         f"main_grad_exists={lora_main_grad_exists}, "
                         f"main_grad_nonzero={lora_main_grad_nonzero}, "
                         f"param.grad_nonzero={lora_grad_nonzero}")

            # Also check the optimizer's model_float16_groups
            if hasattr(self.optimizer, 'model_float16_groups'):
                total_params_in_opt = sum(len(g) for g in self.optimizer.model_float16_groups)
                logger.info(f"[GRAD_DIAG step={step}] optimizer model_float16_groups: "
                             f"{len(self.optimizer.model_float16_groups)} groups, "
                             f"{total_params_in_opt} total params")
            
            # Check if grad_scaler found inf
            if hasattr(self.optimizer, 'grad_scaler') and self.optimizer.grad_scaler is not None:
                logger.info(f"[GRAD_DIAG step={step}] grad_scaler present, "
                             f"loss_scale={self.optimizer.get_loss_scale().item():.4f}")
            else:
                logger.info(f"[GRAD_DIAG step={step}] No grad_scaler (bf16 mode)")

        _diag_step_counter[0] += 1
        return metrics, grad_norm

    trainer.__class__.loss_func = diagnostic_loss_func
    trainer.__class__.train_step = diagnostic_train_step
    logger.info("[GRAD_DIAG] Gradient diagnostics patched successfully")
