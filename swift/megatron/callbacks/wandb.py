# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import time

from swift.utils import check_json_format, is_last_rank
from .base import MegatronCallback
from .utils import rewrite_logs


class WandbCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        args = self.args
        self.config = check_json_format(vars(args))
        if args.wandb_exp_name is None:
            args.wandb_exp_name = args.output_dir
        self.save_dir = os.path.join(args.output_dir, 'wandb')
        self.writer = None
        self.setup()

    def setup(self):
        import wandb
        args = self.args
        if is_last_rank():
            last_exc = None
            for attempt in range(3):
                try:
                    wandb.init(dir=self.save_dir, name=args.wandb_exp_name, project=args.wandb_project, config=self.config)
                except Exception as exc:
                    last_exc = exc
                    print(f'WARN: wandb.init failed on attempt {attempt + 1}/3: {exc}', flush=True)
                    try:
                        wandb.finish(exit_code=1, quiet=True)
                    except Exception:
                        pass
                    time.sleep(5 * (attempt + 1))
                else:
                    self.writer = wandb
                    if wandb.run is not None and getattr(wandb.run, "url", None):
                        print(f'wandb_url: {wandb.run.url}', flush=True)
                    break
            if self.writer is None:
                print(f'WARN: disabling wandb logging after init failures: {last_exc}', flush=True)

    def on_log(self, logs):
        logs = rewrite_logs(logs)
        if is_last_rank() and self.writer is not None:
            self.writer.log(logs, step=self.state.iteration)
