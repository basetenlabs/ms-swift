# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import multiprocessing as mp
import os
import torch.distributed as dist
from itertools import chain
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
from typing import Optional

from swift.template import MaxLengthError
from swift.utils import get_logger, get_object_collective_group, is_dist, is_master, split_list

logger = get_logger()


def calculate_matched_group(sequences, packing_length: int, is_finished: bool = True):
    if len(sequences) == 0:
        return [], []
    # https://arxiv.org/pdf/2404.10830
    import binpacking
    sequences = binpacking.to_constant_volume(sequences, packing_length, weight_pos=1)
    if sequences and not is_finished:
        sequences, ret_sequences = sequences[:-1], sequences[-1]
    else:
        ret_sequences = []
    return sequences, ret_sequences


class PackingDataset(Dataset):
    PACKING_BATCH_SIZE = 1000

    def __init__(
        self,
        template,
        dataset,
        num_proc: int = 1,
        *,
        strict: bool = False,
        load_from_cache_file: bool = True,
        packing_length: Optional[int] = None,
        packing_num_proc: int = 1,
        **kwargs,
    ):
        template.packing = True
        template.padding_free = True  # TODO: remove
        self.template = template
        self.dataset = dataset
        self.num_proc = num_proc
        self.strict = strict
        self.load_from_cache_file = load_from_cache_file
        self.packing_length = packing_length or self.template.max_length
        self.packing_num_proc = min(packing_num_proc, math.ceil(len(dataset) / self.PACKING_BATCH_SIZE))
        self._out_queue = mp.Queue()
        if is_master():
            lengths = self.dataset['lengths']
            offset = 0
            chunked_lengths = split_list(lengths, self.packing_num_proc)
            for i in range(self.packing_num_proc):
                worker = mp.Process(
                    target=self.create_packed_idx, args=(
                        i,
                        offset,
                        chunked_lengths[i],
                    ), daemon=True)
                worker.start()
                offset += len(chunked_lengths[i])
            self.packed_idx = [[] for _ in range(self.packing_num_proc)]
            self.packed_length = [[] for _ in range(self.packing_num_proc)]
            desc = 'Packing: ' if self.packing_num_proc == 1 else f'Packing (num_proc={self.packing_num_proc}): '
            with tqdm(total=len(lengths), dynamic_ncols=True, desc=desc) as prog_bar:
                finished_workers = 0
                while finished_workers < self.packing_num_proc:
                    rank, sequences, data_len = self._out_queue.get()
                    if data_len == -1:
                        finished_workers += 1
                        continue
                    prog_bar.update(data_len)
                    self.packed_idx[rank] += [[x[0] for x in seq] for seq in sequences]
                    self.packed_length[rank] += [sum(x[1] for x in seq) for seq in sequences]
            self.packed_idx = list(chain.from_iterable(self.packed_idx))
            self.packed_length = list(chain.from_iterable(self.packed_length))
        else:
            self.packed_idx, self.packed_length = None, None
        if dist.is_initialized() and is_dist():
            obj_list = [(self.packed_idx, self.packed_length)]
            dist.broadcast_object_list(obj_list, group=get_object_collective_group())
            self.packed_idx, self.packed_length = obj_list[0]

    def create_packed_idx(self, rank, offset, lengths):
        data = [(i + offset, sum(length) if isinstance(length, list) else length) for i, length in enumerate(lengths)]
        i = 0
        input_data = []
        while True:
            new_data = data[i:i + self.PACKING_BATCH_SIZE]
            input_data += new_data
            if not input_data:
                break
            i += self.PACKING_BATCH_SIZE
            is_finished = i >= len(data)
            sequences, input_data = calculate_matched_group(input_data, self.packing_length, is_finished=is_finished)
            self._out_queue.put((rank, sequences, len(new_data)))
        self._out_queue.put((rank, [], -1))

    @staticmethod
    def _actual_row_length(row):
        lengths = []
        for key in ('input_ids', 'labels'):
            value = row.get(key)
            if value is not None:
                lengths.append(len(value))
        if not lengths and row.get('length') is not None:
            length = row['length']
            if isinstance(length, (list, tuple)):
                lengths.append(max(length) if length else 0)
            else:
                lengths.append(length)
        return max(lengths) if lengths else 0

    @staticmethod
    def _slice_row_value(value, length, key):
        if value is None:
            return value
        if hasattr(value, 'dim'):
            if key == 'position_ids' and value.dim() == 3:
                return value[..., :length]
            return value[:length]
        if isinstance(value, tuple):
            return value[:length]
        if isinstance(value, list):
            return value[:length]
        return value

    def _truncate_materialized_row(self, row, length):
        row = dict(row)
        for key in ('input_ids', 'labels', 'loss_scale', 'position_ids'):
            if key in row:
                row[key] = self._slice_row_value(row[key], length, key)
        row['length'] = min(self._actual_row_length(row), length)
        if 'lengths' in row:
            row['lengths'] = [row['length']]
        return row

    def _cap_materialized_rows(self, row, index, sequence):
        """Ensure materialized packed rows do not exceed the configured cap.

        Packing bins are planned from cached per-row lengths and then rows are
        lazily encoded in __getitem__. If those two views ever diverge (for
        example, due to a randomized lazy wrapper), the collator can otherwise
        concatenate a batch longer than packing_length/max_length. Dropping tail
        rows here is preferable to handing an overlong packed sequence to
        Megatron, where it can hang inside the model/pipeline schedule.
        """
        if not self.packing_length or not row:
            return row

        lengths = [self._actual_row_length(r) for r in row]
        total = sum(lengths)
        if total <= self.packing_length:
            return row

        kept, kept_length = [], 0
        for r, length in zip(row, lengths):
            if length > self.packing_length:
                r = self._truncate_materialized_row(r, self.packing_length)
                length = self._actual_row_length(r)
            if kept and kept_length + length > self.packing_length:
                continue
            kept.append(r)
            kept_length += length
            if kept_length >= self.packing_length:
                break

        if os.environ.get('SWIFT_DEBUG_PACKING'):
            planned_length = None
            if self.packed_length is not None and index < len(self.packed_length):
                planned_length = self.packed_length[index]
            logger.warning(
                'Materialized packed sample exceeded packing_length; '
                f'index={index}, sequence={sequence}, planned_length={planned_length}, '
                f'actual_lengths={lengths}, actual_total={total}, '
                f'kept_count={len(kept)}, kept_total={kept_length}, '
                f'packing_length={self.packing_length}.')

        return kept or row[:1]

    def __getitem__(self, index):
        sequence = self.packed_idx[index]
        row = [self.dataset[i] for i in sequence]
        row = self._cap_materialized_rows(row, index, sequence)
        return row

    def __len__(self):
        return len(self.packed_idx)


class IterablePackingDataset(IterableDataset):

    def __init__(
        self,
        template,
        dataset,
        num_proc: int = 1,
        *,
        packing_interval: int = 128,
        packing_length: Optional[int] = None,
        strict: bool = False,
        cyclic: bool = False,
        **kwargs,
    ):
        template.packing = True
        template.padding_free = True  # TODO: remove
        self.template = template
        self.dataset = dataset
        self.num_proc = num_proc
        self.strict = strict
        self.packing_length = packing_length or self.template.max_length

        self.packing_interval = packing_interval
        self._in_queue = mp.Queue()
        self._out_queue = mp.Queue()
        self.workers = []
        self.cyclic = cyclic
        for _ in range(self.num_proc):
            worker = mp.Process(target=self._processor, daemon=True)
            worker.start()
            self.workers.append(worker)

    def _processor(self):
        while True:
            i, data = self._in_queue.get()
            encoded_data = {}
            try:
                encoded_data = self.template.encode(data, return_length=True)
            except Exception as e:
                if self.strict and not isinstance(e, MaxLengthError):
                    raise
            self._out_queue.put((i, encoded_data))

    def _put_data_in_queue(self, iterator) -> int:
        for i in range(self.packing_interval):
            try:
                data = next(iterator)
            except StopIteration:
                return i
            self._in_queue.put((i, data))
        return i + 1

    def _fetch_data_out_queue(self, last_res, num_samples):
        res = [None] * num_samples
        for _ in range(num_samples):
            i, data = self._out_queue.get()
            if not data:
                continue
            res[i] = (data, len(data['input_ids']))
        res = [data for data in res if data]
        last_res += res
        return last_res

    @staticmethod
    def cyclic_iter(iterable):
        while True:
            for x in iterable:
                yield x

    def __iter__(self):
        try:
            next(iter(self.dataset))
        except StopIteration:
            return

        if self.cyclic:
            iterator = self.cyclic_iter(self.dataset)
        else:
            iterator = iter(self.dataset)
        data = []
        while True:
            num_samples = self._put_data_in_queue(iterator)
            finished = num_samples != self.packing_interval
            data = self._fetch_data_out_queue(data, num_samples)
            sequences, data = calculate_matched_group(data, self.packing_length, is_finished=finished)
            res = []
            for row in sequences:
                res.append([r[0] for r in row])
            yield from res
            if finished:
                break
