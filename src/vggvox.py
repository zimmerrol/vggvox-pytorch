# Copyright (c) 2018 Roland Zimmermann
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.modules.normalization as normalization

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ShapePrinter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x

class VGGVox(nn.Module):
    def __init__(self, n, num_classes=1000, aux_logits=False):
        super().__init__()

        if type(n) is not int or n < 1:
            raise ValueError('num_classes must be a positive integer.')
        self._n = n

        if num_classes is not None and (type(num_classes) is not int or num_classes < 1):
            raise ValueError('num_classes must be a positive integer.')
        self._num_classes = num_classes

        self._aux_logits = aux_logits

        if self._num_classes is None and not self._aux_logits:
            raise ValueError('Either num_classes must be set to a positive integer or aux_logits must be True')

        # self._logits = logits
        # if self._logits and not self._num_classes:
        #     raise ValueError("Cannot return logits without specifying num_classes")

        self._feature_generator = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(7, 7), stride=(2, 2), padding=1),
            nn.ReLU(),
            normalization.LocalResponseNorm(size=5, k=2),

            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            
            nn.BatchNorm2d(96),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=1),
            nn.ReLU(),
            normalization.LocalResponseNorm(size=5, k=2),

            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),

            nn.BatchNorm2d(384),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),

            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2)),

            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(9, 1), stride=(1, 1)),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=(1, self._n), stride=1),
            Flatten(),
        )

        if self._num_classes:
            # add classification head
            self._classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=4096, out_features=1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=1024, out_features=num_classes),
                nn.Softmax()
            )

    def forward(self, x):
        x_aux_logits = self._feature_generator(x)

        if self._num_classes:
            # calculate forward pass of the classification head
            y = self._classifier(x_aux_logits)

        if self._num_classes:
            if self._aux_logits:
                return x_aux_logits, y
            else:
                return y
        else:
            if self._aux_logits:
                return x_aux_logits

class VGGVoxRegression(VGGVox):
    def __init__(self, n, num_values, activation=None):
        super().__init__(n=n, num_classes=None, aux_logits=True)
        self._activation = activation
        self._regressor = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=4096, out_features=1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=1024, out_features=num_values),
            )

    def forward(self, x):
        x = super().forward(x)
        x = self._regressor(x)
        if self._activation:
            x = self._activation(x)

        return x