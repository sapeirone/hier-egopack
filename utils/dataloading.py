import torch
import random
import numpy as np

import os
import torch.utils.data
from torch.utils.data.dataloader import default_collate

from typing import Tuple, Union, Dict, List

from torch_geometric.loader import DataLoader


class multiloader:
    def __init__(self, loaders, weights):
        self.loaders = loaders
        self.weights = weights
        # create an iterator for each loader
        self.iterators = [
            iter(loader) if loader is not None and weight > 0 else None
            for loader, weight in zip(self.loaders, self.weights)
        ]

        # mark absent iterator as already completed
        self.completed = [iterator is None for iterator in self.iterators]

    def __iter__(self):
        return self
    
    def __len__(self):
        return max(
            len(loader) if loader is not None and weight > 0 else 0
            for loader, weight in zip(self.loaders, self.weights)
        )

    def __next__(self):
        data = []

        # iterate over all iterators
        for i in range(len(self.loaders)):
            if self.iterators[i] is None:
                data.append(None)
                continue

            try:
                # try loading from this iterator
                data.append(next(self.iterators[i]))  # type: ignore
            except StopIteration:
                # this iterator is exhausted -> mark it as completed
                self.completed[i] = True
                if all(self.completed):
                    # all iterators are exhausted -> we are done
                    raise StopIteration

                # some iterators still need to complete -> reset this iterator and keep looping
                self.iterators[i] = iter(self.loaders[i])
                data.append(next(self.iterators[i]))  # type: ignore

        return tuple(data)
    

class InfiniteLoader:
    def __init__(self, loaders: Union[List[DataLoader], Dict[str, DataLoader]]):
        if isinstance(loaders, list):
            loaders = {str(i): loader for i, loader in enumerate(loaders)}
            self.return_as_tuple = True
        else:
            self.return_as_tuple = False
            
        self.loaders = loaders
        # create an iterator for each loader
        self.iterators = {name: iter(loader) for name, loader in self.loaders.items()}

    def __iter__(self):
        return self
    
    def __len__(self):
        return max(len(loader) for loader in self.loaders.values())

    def __next__(self):
        data = dict()

        # iterate over all iterators
        for name, iterator in self.iterators.items():

            try:
                # try loading from this iterator
                data[name] = next(iterator)  # type: ignore
            except StopIteration:
                # some iterators still need to complete -> reset this iterator and keep looping
                self.iterators[name] = iter(self.loaders[name])
                data[name] = next(self.iterators[name])  # type: ignore

        if self.return_as_tuple:
            return tuple(data.values())
        
        return data


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_dataloader(dataset, batch_size, is_training, num_workers, drop_last, rng_generator=None):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        generator=rng_generator,
        follow_batch=['labels', 'segments', 'narration_timestamps', 'extra_narrations']
    )
