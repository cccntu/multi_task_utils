import random
from enum import Enum

import pytorch_lightning as pl
from torch.utils.data import IterableDataset


class TaskSamplingStrategy(Enum):
    propotional = 0
    union = 0
    equal = 1
    none = 2
    round_robin = 3
    parallel = 4


def unlimited(iterator):
    while True:
        for x in iterator:
            yield x


class MultiTaskDataLoader(IterableDataset):
    """wraps dataloaders into one dataloader"""

    def __init__(self, dataloaders, strategy: TaskSamplingStrategy.none, seed=42):
        self.dataloaders = dataloaders
        self.strategy = strategy
        self.lens = [len(dl) for dl in self.dataloaders]
        self.rng = random.Random(seed)
        self.dataset = [0 for _ in range(len(self))]

    def __len__(self):
        if self.strategy == TaskSamplingStrategy.parallel:
            return min(self.lens)
        else:
            return sum(self.lens)

    def __iter__(self):
        if self.strategy == TaskSamplingStrategy.union:
            iterators = [umlimited(dl) for dl in self.dataloaders]
            ids = []
            for i, dl in enumerate(self.dataloaders):
                ids.extend([i] * len(dl))
            self.rng.shuffle(ids)
            for id in ids:
                nxt = next(iterators[id])
                yield nxt
        elif self.strategy == TaskSamplingStrategy.propotional:
            iterators = [umlimited(dl) for dl in self.dataloaders]
            ids = list(range(len(iterators)))
            weights = self.lens
            for id in self.rng.choices(ids, weights=weights, k=len(self)):
                nxt = next(iterators[id])
                yield nxt
        elif self.strategy == TaskSamplingStrategy.equal:
            iterators = [umlimited(dl) for dl in self.dataloaders]
            ids = list(range(len(iterators)))
            for id in self.rng.choices(ids, k=len(self)):
                nxt = next(iterators[id])
                yield nxt
        elif self.strategy == TaskSamplingStrategy.round_robin:
            iterators = [umlimited(dl) for dl in self.dataloaders]
            n = len(self.dataloaders)
            for i in range(len(self)):
                id = i % n
                nxt = next(iterators[id])
                yield nxt

        elif self.strategy == TaskSamplingStrategy.none:
            for dl in self.dataloaders:
                for x in dl:
                    yield x
        elif self.strategy == TaskSamplingStrategy.parallel:
            # return iter(zip(*self.dataloaders))
            for batches in zip(*self.dataloaders):  # self.dataloaders:
                yield batches
                # for x in dl:
                #    yield x
            # return zip(*self.dataloaders)
        else:
            NotImplementedError()


class MultiTaskDataModule(pl.LightningDataModule):
    """wraps multiple pl.LightningDataModule into one"""

    def __init__(self, datamodules, strategy=TaskSamplingStrategy.none):
        super().__init__()
        self.datamodules = datamodules
        self.strategy = strategy

    def prepare_data(self):
        for datamodule in self.datamodules:
            datamodule.prepare_data()

    def setup(self, stage=None):
        for datamodule in self.datamodules:
            datamodule.setup(stage)

    def train_dataloader(self):
        dataloaders = [datamodule.train_dataloader() for datamodule in self.datamodules]
        return MultiTaskDataLoader(dataloaders=dataloaders, strategy=self.strategy)

    def val_dataloader(self):
        dataloaders = [datamodule.val_dataloader() for datamodule in self.datamodules]
        dataloaders = [x for x in dataloaders if x is not None]
        return MultiTaskDataLoader(
            dataloaders=dataloaders, strategy=TaskSamplingStrategy.none
        )


#     def test_dataloader(self):
#         return DataLoader(self.testset, batch_size=8, collate_fn=self.data_collator)
