# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle.io import Dataset, RandomSampler
import configs
import numpy as np


class DataLoader(paddle.io.DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=2,
                 collate_fn=None,
                 pin_memory=False,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None,
                 multiprocessing_context=None,
                 generator=None):
        if isinstance(dataset[0], (tuple, list)):
            return_list = True
        else:
            return_list = False
        return_list = True
        super().__init__(
            dataset,
            feed_list=None,
            places=None,
            return_list=return_list,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=num_workers,
            use_buffer_reader=True,
            use_shared_memory=True,
            timeout=timeout,
            worker_init_fn=worker_init_fn)
        if sampler is not None:
            self.batch_sampler.sampler = sampler


def get_data_labels(dataset: Dataset):
    """ split dataset into tensor of data inputs and thier labels"""
    inputs = []
    targets = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        inputs.append(x)
        targets.append(y)

    inputs = paddle.stack(inputs)
    targets = paddle.to_tensor(targets)
    return inputs, targets


def get_train_val_dls(dataset: Dataset, batch_size):
    """
    same as previous function without test (use this function in case that test is already separated).
    all inputs as in prev function (i.e. get_train_val_test_dls).
    :return: train & val data loaders of type DataLoader.
    """

    ds_size = len(dataset)
    indices = list(range(ds_size))
    np.random.shuffle(indices)
    val_split_idx = int(ds_size * configs.val_ratio)
    train_indices, val_indices = indices[:val_split_idx], indices[val_split_idx:]

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=RandomSampler(train_indices))
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=RandomSampler(val_indices))

    return train_loader, validation_loader


def dataset_to_dataloader(dataset: Dataset, batch_size):
    """ A function that gets dataset and returns simple data loader.
        It is useful when the dataset comes slitted (to train and test)"""

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
