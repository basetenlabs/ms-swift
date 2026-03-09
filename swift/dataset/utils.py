# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import numpy as np
import os
import tempfile
from datasets import Dataset as HfDataset
from modelscope.hub.utils.utils import get_cache_dir
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, Optional, Union

from swift.template import Template
from swift.utils import get_logger
from .preprocessor import RowPreprocessor

logger = get_logger()


def sample_dataset(
        dataset: HfDataset,
        dataset_sample: Optional[int],
        shuffle: bool = True,
        random_state: Optional[np.random.RandomState] = None,
        shuffle_all: bool = False,  # For compatibility, this defaults to False.
) -> HfDataset:
    """Sample dataset by a dataset_sample number
    Args:
        dataset: The dataset instance, iterable dataset is not supported
        dataset_sample: The sample number
        shuffle: Whether to perform random sampling on non-streaming datasets
        random_state: The random state
    Returns:
        The sampled dataset
    """
    if dataset_sample is None:
        return dataset

    n_repeat_sample = dataset_sample // len(dataset)
    n_remain_sample = dataset_sample % len(dataset)
    if n_repeat_sample >= 1 and n_remain_sample >= 1:
        logger.warning(f'dataset_sample:{dataset_sample} is greater than len(dataset):{len(dataset)}, '
                       'repeated sampling will be performed.')
    idx = np.tile(range(len(dataset)), n_repeat_sample)
    if random_state is None:
        random_state = np.random.RandomState()
    if n_remain_sample >= 1:
        if shuffle:
            idx_remain = random_state.permutation(len(dataset))[:n_remain_sample]
        else:
            idx_remain = np.arange(n_remain_sample)
        idx = np.concatenate([idx, idx_remain])
    if n_repeat_sample >= 1 and shuffle and shuffle_all:
        random_state.shuffle(idx)
    dataset = dataset.select(idx)
    return dataset


class LazyLLMDataset(Dataset):
    """This class if used to lazy tokenize the dataset, and skips bad ones when training"""

    def __init__(self,
                 dataset: HfDataset,
                 encode_func: Callable[[Dict[str, Any]], Dict[str, Any]],
                 *,
                 n_try_fetch: int = 10,
                 strict: bool = False,
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 traceback_limit: int = 10) -> None:
        self.dataset = dataset
        self.encode_func = encode_func

        n_try_fetch = 1 if strict else min(n_try_fetch, len(self.dataset))
        assert n_try_fetch >= 1
        self.strict = strict
        self.n_try_fetch = n_try_fetch

        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state

        self.traceback_limit = traceback_limit
        self._traceback_counter = 0
        self._idx = 0
        self._idx_list = self.random_state.permutation(len(self.dataset)).tolist()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if isinstance(idx, str):
            return self.dataset[idx]
        for i in range(self.n_try_fetch):
            n_try = i
            if i == 0:
                i = idx
            else:
                i = self._idx_list[self._idx]
                self._idx = (self._idx + 1) % len(self.dataset)
            data = self.dataset[i]
            try:
                return self.encode_func(data, return_length=True)
            except Exception:
                if n_try == self.n_try_fetch - 1 or self.strict:
                    if self.strict:
                        logger.warning('To avoid errors, you can pass `strict=False`.')
                    raise
                if self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
                    import traceback
                    logger.info(traceback.format_exc())
                    logger.warning('👆👆👆There are errors in the template.encode, '
                                   'and another piece of data will be randomly selected.')
                    self._traceback_counter += 1

        raise ValueError('Failed to retrieve the dataset. You can avoid this issue by increasing `max_length` or '
                         'modifying the `truncation_strategy`.')

    def __len__(self) -> int:
        return len(self.dataset)


class SequentialSkipLazyLLMDataset(LazyLLMDataset):
    """Resilient lazy dataset with optional randomized traversal and permanent over-length skipping."""

    def __init__(self,
                 dataset: HfDataset,
                 encode_func: Callable[[Dict[str, Any]], Dict[str, Any]],
                 *,
                 randomize_dataset: bool = False,
                 random_seed: Optional[int] = None,
                 traceback_limit: int = 10) -> None:
        # strict is intentionally disabled for this variant.
        super().__init__(
            dataset,
            encode_func,
            n_try_fetch=max(1, len(dataset)),
            strict=False,
            random_state=random_seed,
            traceback_limit=traceback_limit)
        # Behave like "infinite n_try_fetch": keep retrying until a valid item is found.
        self.n_try_fetch = float('inf')
        self.randomize_dataset = randomize_dataset
        self._skip_idx_set = set()
        self._repeat_traceback_counts = {}
        if self.randomize_dataset:
            self._idx_order = self.random_state.permutation(len(self.dataset)).tolist()
        else:
            self._idx_order = list(range(len(self.dataset)))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if isinstance(idx, str):
            return self.dataset[idx]
        dataset_len = len(self.dataset)
        start_pos = idx % dataset_len
        offset = 0

        while True:
            i = self._idx_order[(start_pos + offset) % dataset_len]
            offset += 1
            if i in self._skip_idx_set:
                continue

            data = self.dataset[i]
            try:
                return self.encode_func(data, return_length=True)
            except Exception as e:
                try:
                    from swift.template import MaxLengthError
                    if isinstance(e, MaxLengthError):
                        self._skip_idx_set.add(i)
                        continue
                except Exception:
                    pass
                self._log_traceback_with_dedupe(e)
                continue

    def _log_traceback_with_dedupe(self, e: Exception) -> None:
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            frame = tb[-1]
            site = (frame.filename, frame.name, frame.lineno, type(e).__name__)
        else:
            site = ("<unknown>", "<unknown>", -1, type(e).__name__)

        seen_count = self._repeat_traceback_counts.get(site, 0) + 1
        self._repeat_traceback_counts[site] = seen_count

        if seen_count == 1 and (self.traceback_limit is None or self._traceback_counter < self.traceback_limit):
            logger.info(''.join(traceback.format_exception(type(e), e, e.__traceback__)))
            logger.warning('👆👆👆There are errors in the template.encode; selecting another sample.')
            self._traceback_counter += 1
            return

        logger.warning(
            f'Similar traceback seen {seen_count} times at '
            f'{site[0]}:{site[2]} ({site[1]}::{site[3]}). Continuing with another sample.')


class EncodePreprocessor(RowPreprocessor):

    def __init__(self, template: 'Template'):
        super().__init__()
        self.template = template

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.template.encode(row, return_length=True)


class AddLengthPreprocessor(EncodePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        encoded = super().preprocess(row)
        row['lengths'] = encoded['lengths']
        return row


TEMP_DIR_POOL = {}


def get_temporary_cache_files_directory(prefix=None):
    if prefix is None:
        import datasets.config
        prefix = datasets.config.TEMP_CACHE_DIR_PREFIX
    if prefix in TEMP_DIR_POOL:
        TEMP_DIR = TEMP_DIR_POOL[prefix]
    else:
        tmp_dir = os.path.join(get_cache_dir(), 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        kwargs = {}
        parameters = inspect.signature(tempfile.TemporaryDirectory.__init__).parameters
        if 'ignore_cleanup_errors' in parameters:
            kwargs['ignore_cleanup_errors'] = True
        TEMP_DIR = tempfile.TemporaryDirectory(prefix=prefix, dir=tmp_dir, **kwargs)
        logger.info(f'create tmp_dir: {TEMP_DIR.name}')
        TEMP_DIR_POOL[prefix] = TEMP_DIR

    return TEMP_DIR.name
