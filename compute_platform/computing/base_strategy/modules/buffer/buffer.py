import numpy as np
import torch

from computing.core.dtype import LearningVariable, BufferParameter
from computing.core.module import Module

class Buffer(Module):
    """
    Buffer基类，需要指定buffer_size和首先buffer更新函数
    """
    def __init__(self, buffer_size=128, module_id=-1, **kwargs):
        """
        Args:
            buffer_size: buffer支持的最大容量
            current_load_num: 记录buffer目前已经装载的数量
            is_full: buffer是否装满
        """
        super(Buffer, self).__init__(module_id=module_id)
        self.buffer_size = BufferParameter(buffer_size)
        self.current_load_num = BufferParameter(0)
        self.is_full = BufferParameter(False)

    def update(self, *args, **kwargs):
        raise  NotImplementedError("A Buffer module must have a update function")


class TensorAndListBuffer(Buffer):
    """
    支持装载tensor和list的buffer
    """
    def __init__(self, buffer_size=128, module_id=-1, **kwargs):
        super(TensorAndListBuffer, self).__init__(module_id=module_id, buffer_size=buffer_size)
        self.register_decide_hooks(["is_full"])

    def update(self, new_data: dict=None, release=False, *args, **kwargs):
        """
        Args:
            new_data: dict, key是被装载的数据名字，value是被装载的数据
            release: 是否清空buffer。当release是ture时，且new_data is not None时，会先清空buffer再装载
        Note: 装载和清空必须至少做一个事，即不能同时new_data is None且release is False
        """
        assert new_data is not None or release is not False
        if type(new_data) is not dict: new_data = None

        if release:
            self._release()

        if new_data is not None:
            for key, value in new_data.items():
                if key not in self.decide().keys():
                    setattr(self, key, BufferParameter(None))
                    self.register_decide_hooks([key])

                if getattr(self, key) is None:
                    setattr(self, key, value)
                else:
                    buffer_value = getattr(self, key)
                    if type(value) is torch.Tensor:
                        start_index = 1 if self.is_full else 0
                        buffer_value = torch.cat([buffer_value[start_index:], value.detach()])
                        setattr(self, key, buffer_value)
                    elif type(value) is list:
                        if self.is_full: buffer_value.pop(0)
                        if type(value[0]) is torch.Tensor: value = value[0].detach()
                        buffer_value.append(value)
                    setattr(self, key, buffer_value)

            self.current_load_num = min(self.current_load_num + 1, self.buffer_size)
            if self.current_load_num == self.buffer_size:
                self.is_full = True

    def _release(self):
        for key in self.decide().keys():
            if key!="is_full": setattr(self, key, None)
        self.current_load_num = BufferParameter(0)
        self.is_full = BufferParameter(False)


