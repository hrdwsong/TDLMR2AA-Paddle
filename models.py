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

import paddle.nn as nn


class CNN_MNIST_OriginNet(nn.Layer):
    name = "MNIST-ORIGIN-NET"

    def __init__(self):
        super().__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.KaimingUniform())
        self.cnn = nn.Sequential(
            nn.Conv2D(1, 32, 5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),  # 28 -> 14
            nn.Conv2D(32, 64, 5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),  # 14 -> 7

        )

        self.lin = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape([-1, 64*7*7])
        x = self.lin(x)
        return x


class CNN_MNIST_B(nn.Layer):
    name = "MNIST-B-NET"

    def __init__(self):
        super().__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.KaimingUniform())
        self.cnn = nn.Sequential(
            nn.Dropout(p=0.2),
            # nn.Conv2D(1, 64, 8, padding='same', stride=1),
            nn.Conv2D(1, 64, 8, stride=1),
            nn.ReLU(),
            # nn.MaxPool2D(kernel_size=2, stride=2),  # 28 -> 14
            # nn.Conv2D(64, 128, 6, padding='same', stride=1),
            nn.Conv2D(64, 128, 6, stride=1),
            nn.ReLU(),
            # nn.MaxPool2D(kernel_size=2, stride=2),  # 14 -> 7
            # nn.Conv2D(128, 128, 5, padding='same', stride=1),
            nn.Conv2D(128, 128, 5, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.lin = nn.Sequential(
            # nn.Linear(7*7*128, 10)
            nn.Linear(12 * 12 * 128, 10)
        )

    def forward(self, x):
        x = self.cnn(x)
        # x = x.reshape([-1, 7*7*128])
        x = x.reshape([-1, 12 * 12 * 128])
        x = self.lin(x)
        return x
