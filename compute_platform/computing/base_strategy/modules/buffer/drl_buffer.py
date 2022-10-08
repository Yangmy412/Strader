import numpy as np
import torch

from computing.core.dtype import LearningVariable, BufferParameter
from computing.core.module import Module
from computing.base_strategy.modules.buffer.buffer import Buffer


class DRLBuffer(Buffer):
    """
    DRLBuffer class, which is a data structure of buffer for batch training
     "X": input state data [batch, feature, stock];
     "y": future relative price [batch, norm_feature, coin];
     "last_w:" a tensor with shape [batch_size, assets];
    """

    def __init__(self, buffer_size=128, sample_size=64, module_id=-1, transaction_cost=0.025, **kwargs):
        super(DRLBuffer, self).__init__(buffer_size=buffer_size, sample_size=sample_size, module_id=module_id)
        self.sample_size = BufferParameter(sample_size)

        self.buffer_state = BufferParameter(None)
        self.buffer_reward = BufferParameter(None)
        self.buffer_action = BufferParameter(None)
        self.sample_state = BufferParameter(None)
        self.sample_reward = BufferParameter(None)
        self.sample_action = BufferParameter(None)
        self.transaction_cost = BufferParameter(torch.tensor(transaction_cost))

        self.register_decide_hooks(["sample_state", "sample_reward", "sample_action",
                                    "buffer_state", "buffer_reward", "buffer_action",
                                    "is_full", "transaction_cost"])

        self.acounter = BufferParameter(0)


    def update(self, state=None, action=None, reward=None, release=False, *args, **kwargs):
        if release:
            self._release()
        else:
            self.acounter = min(self.acounter+1, self.buffer_size)
            self.buffer_state = self._update_single_data(self.buffer_state, state)
            self.buffer_action = self._update_single_data(self.buffer_action, action)
            self.buffer_reward = self._update_single_data(self.buffer_reward, reward)

            sample_index = self._sample()
            if sample_index is not None:
                self.sample_state = self._sample_single_data(self.buffer_state, sample_index)
                self.sample_reward = self._sample_single_data(self.buffer_reward, sample_index)
                self.sample_action = self._sample_single_data(self.buffer_action, sample_index)

    # ================================== update function ==================================
    def _update_single_data(self, buffer_x, x):
        if buffer_x is None:
            return x

        data_type = type(x)
        if data_type is torch.Tensor:
            start_index = 1 if buffer_x.shape[0] == self.buffer_size else 0
            buffer_x = torch.cat([buffer_x[start_index:], x])
        elif data_type == list:
            if len(buffer_x) == self.buffer_size:
                buffer_x.pop(0)
            buffer_x.append(x[0])
        elif data_type == dict:
            for key, value in x.items():
                buffer_x[key] = self._update_single_data(buffer_x[key], value)
        else:
            raise IOError("Please check element type, which only supports tensor, list and dict.")

        return buffer_x

    def _release(self):
        print("the buffer has been released.")
        self.buffer_state = BufferParameter(None)
        self.buffer_reward = BufferParameter(None)
        self.buffer_action = BufferParameter(None)
        self.sample_state = BufferParameter(None)
        self.sample_reward = BufferParameter(None)
        self.sample_action = BufferParameter(None)
        self.is_full = BufferParameter(False)
        self.acounter = BufferParameter(0)


    # ======================================================================================

    # ================================== sample function ==================================
    def _get_sample_indices(self):
        sample_bias = 0.1
        ran = np.random.geometric(sample_bias)
        while ran > self.acounter - self.sample_size:
            ran = np.random.geometric(sample_bias)
        sample_indices = range(ran, ran + self.sample_size)
        return sample_indices

    def _sample(self):
        if self.acounter < self.sample_size:
            return None
        self.is_full = True

        if self.acounter - self.sample_size == 0:
            return range(0, self.sample_size)
        sample_indices = self._get_sample_indices()
        return sample_indices

    def _sample_single_data(self, buffer_x, sample_index):
        if buffer_x is None:
            return None

        data_type = type(buffer_x)
        if data_type is torch.Tensor:
            sample_x = buffer_x[sample_index]
        elif data_type == list:
            sample_x = [buffer_x[index] for index in sample_index]
        elif data_type == dict:
            sample_x = {}
            for key, value in buffer_x.items():
                sample_x[key] = self._sample_single_data(buffer_x[key], sample_index)
        else:
            raise IOError("Please check element type, which only supports tensor, list and dict.")
        return sample_x

    # ======================================================================================





class PROFITBuffer(DRLBuffer):
    def __init__(self, buffer_size=8, sample_size=4, transaction_cost=0.025, **kwargs):
        super(PROFITBuffer, self).__init__(sample_size=sample_size, transaction_cost=transaction_cost)
        self.buffer_size = buffer_size
        self.buffer_weight_last = BufferParameter(None)
        self.buffer_weight = BufferParameter(None)
        self.buffer_action = BufferParameter(None)
        self.buffer_reward = BufferParameter(None)
        self.buffer_state = BufferParameter(None)
        self.buffer_state_next = BufferParameter(None)

        self.sample_weight_last = BufferParameter(None)
        self.sample_weight = BufferParameter(None)
        self.sample_action = BufferParameter(None)
        self.sample_reward = BufferParameter(None)
        self.sample_state = BufferParameter(None)
        self.sample_state_next = BufferParameter(None)

        self.is_full = BufferParameter(False)
        self.register_decide_hooks(["buffer_weight", "sample_weight_last", "sample_weight", "sample_action",
                                    "sample_reward", "sample_state", "sample_state_next", "is_full",
                                    "transaction_cost"])

    def sample(self):
        if self.buffer_reward.shape[0] < self.sample_size:
            return None

        self.is_full = True
        sample_index = torch.randperm(len(self.buffer_reward))[:self.sample_size].sort().values
        return sample_index

    def update(self, weight_last, weight, action, reward, state, state_next, *args, **kwargs):
        if reward is not None:
            if self.buffer_reward is None:
                self.buffer_weight_last = weight_last
                self.buffer_weight = weight
                self.buffer_action = action
                self.buffer_reward = reward
                self.buffer_state = state
            elif self.buffer_reward.shape[0] == self.buffer_size:
                self.buffer_weight_last = torch.cat([self.buffer_weight_last[1:], weight_last])
                self.buffer_weight = torch.cat([self.buffer_weight[1:], weight])
                self.buffer_action = torch.cat([self.buffer_action[1:], action])
                self.buffer_reward = torch.cat([self.buffer_reward[1:], reward])
                for key in state.keys():
                    self.buffer_state[key].pop(0)
                    self.buffer_state[key].append(state[key][0])

            else:
                self.buffer_weight_last = torch.cat([self.buffer_weight_last, weight_last])
                self.buffer_weight = torch.cat([self.buffer_weight, weight])
                self.buffer_action = torch.cat([self.buffer_action, action])
                self.buffer_reward = torch.cat([self.buffer_reward, reward])
                for key in state.keys():
                    self.buffer_state[key].append(state[key][0])
        if state_next is not None:
            if self.buffer_state_next is None:
                self.buffer_state_next = state_next
            elif len(self.buffer_state_next["news"]) == self.buffer_size:
                for key in state_next.keys():
                    self.buffer_state_next[key].pop(0)
                    self.buffer_state_next[key].append(state_next[key][0])
            else:
                for key in state_next.keys():
                    self.buffer_state_next[key].append(state_next[key][0])

            sample_index = self._sample()
            if sample_index is not None:
                self.sample_weight_last = self.buffer_weight_last[sample_index]
                self.sample_weight = self.buffer_weight[sample_index]
                self.sample_action = self.buffer_action[sample_index]
                self.sample_reward = self.buffer_reward[sample_index]

                self.sample_state = {}
                self.sample_state_next = {}
                for key in self.buffer_state.keys():
                    self.sample_state[key] = [self.buffer_state[key][i] for i in sample_index]
                    self.sample_state_next[key] = [self.buffer_state_next[key][i] for i in sample_index]


