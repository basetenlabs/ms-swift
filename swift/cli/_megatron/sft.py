# Copyright (c) ModelScope Contributors. All rights reserved.
import os


def _enable_whetstone_faulthandler() -> None:
    """Enable stack dumps for long-running distributed smoke jobs.

    When a rank wedges inside a second forward/backward step, ptrace-based tools
    (py-spy/gdb) are usually blocked by the container security profile.  Keeping
    faulthandler armed lets us send SIGUSR1 to the Python workers and get Python
    stacks in the Baseten logs without attaching a debugger.
    """
    if os.environ.get('WHETSTONE_FAULTHANDLER', '1').strip().lower() in {'0', 'false', 'no', 'off'}:
        return
    try:
        import faulthandler
        import signal
        import sys

        faulthandler.enable(file=sys.stderr, all_threads=True)
        if hasattr(signal, 'SIGUSR1'):
            try:
                faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
            except RuntimeError:
                # Already registered by a parent import path.
                pass
    except Exception:
        # Debug aid only: never block training startup because signal stack
        # dumping is unavailable on a platform.
        pass


if __name__ == '__main__':
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    _enable_whetstone_faulthandler()
    from swift.megatron import megatron_sft_main
    megatron_sft_main()
