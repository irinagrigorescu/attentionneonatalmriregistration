# ============================================================================
#
#    Copyright 2020-2023 Irina Grigorescu
#    Copyright 2020-2023 King's College London
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ============================================================================

##############################################################################
#
# schedulers.py
#
##############################################################################


class LambdaLR():
    """
    Link: https://github.com/arnab39/cycleGAN-PyTorch/blob/master/utils.py
    """
    def __init__(self, epochs, offset, decay_epoch):
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch) / \
               (self.epochs - self.decay_epoch)
