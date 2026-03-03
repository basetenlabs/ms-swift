# Copyright (c) ModelScope Contributors. All rights reserved.
import time

import torch
from tqdm import tqdm

from swift.megatron.utils import reduce_max_stat_across_model_parallel_group
from swift.utils import format_time, get_logger, is_last_rank
from .base import MegatronCallback

logger = get_logger()


class PrintCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.training_bar = None
        self.eval_bar = None
        self.is_write_rank = is_last_rank()
        self.train_metric_sums = {}
        self.train_metric_count = 0.0

    def on_train_begin(self):
        self.training_bar = tqdm(
            total=self.args.train_iters, dynamic_ncols=True, disable=not self.is_write_rank, desc='Train: ')
        self.start_step = self.state.iteration
        self.training_bar.update(self.state.iteration)
        self.current_step = self.state.iteration
        self.start_time = time.time()
        self.last_log_step = self.state.iteration
        self.last_log_time = self.start_time
        self.train_metric_sums = {}
        self.train_metric_count = 0.0

    def on_train_end(self):
        if self.is_write_rank and self.train_metric_count > 0:
            avg_metrics = {
                key: value / self.train_metric_count for key, value in self.train_metric_sums.items()
            }
            avg_metrics = {k: round(v, 8) for k, v in sorted(avg_metrics.items())}
            self.training_bar.write(
                f"train_avg_metrics(samples={int(self.train_metric_count)}): {avg_metrics}"
            )
        self.training_bar.close()
        self.training_bar = None

    def on_step_end(self):
        n_step = self.state.iteration - self.current_step
        self.current_step = self.state.iteration
        self.training_bar.update(n_step)

    def on_eval_begin(self):
        self.eval_bar = tqdm(
            total=self.args.eval_iters, dynamic_ncols=True, disable=not self.is_write_rank, desc='Evaluate: ')

    def on_eval_end(self):
        self.eval_bar.close()
        self.eval_bar = None

    def on_eval_step(self):
        self.eval_bar.update()

    def on_log(self, logs):
        state = self.state
        args = self.args
        is_eval_log = any(k.startswith('eval_') for k in logs.keys())
        logs['iteration'] = f'{state.iteration}/{args.train_iters}'
        elapsed = time.time() - self.start_time
        logs['elapsed_time'] = format_time(elapsed)
        window_steps = state.iteration - self.last_log_step
        window_elapsed = time.time() - self.last_log_time
        train_speed = logs.get('train_speed(s/it)')
        if not isinstance(train_speed, (int, float)):
            train_speed = window_elapsed / window_steps if window_steps > 0 else 0.0
        train_speed = float(train_speed)
        logs['remaining_time'] = format_time((args.train_iters - state.iteration) * train_speed)
        memory = reduce_max_stat_across_model_parallel_group(torch.cuda.max_memory_reserved() / 1024**3)
        logs['memory(GiB)'] = round(memory, 2)
        logs['train_speed(s/it)'] = round(train_speed, 6)
        sample_weight = float(window_steps) if window_steps > 0 else 1.0
        if not is_eval_log:
            self._update_running_average(logs, sample_weight)
        if not is_eval_log and window_steps > 0:
            self.last_log_step = state.iteration
            self.last_log_time = time.time()
        logs = {k: round(v, 8) if isinstance(v, float) else v for k, v in logs.items()}
        if self.is_write_rank:
            self.training_bar.write(str(logs))

    def _update_running_average(self, logs, sample_weight: float) -> None:
        self.train_metric_count += sample_weight
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                if key not in self.train_metric_sums:
                    self.train_metric_sums[key] = 0.0
                self.train_metric_sums[key] += float(value) * sample_weight
