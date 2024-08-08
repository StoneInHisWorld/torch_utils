from functools import wraps

import torch


class hook:

    def __init__(self):
        self.__last_forward_output, self.__last_backward_data = {}, {}
        self.trainer = None
        self.with_hook = False
        self.hook_mute = True

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.trainer = args[0]
            self.with_hook = self.trainer.runtime_cfg['with_hook']
            self.hook_mute = self.trainer.runtime_cfg['hook_mute']
            if self.with_hook:
                __f_handles, __b_handles = self.__deal_with_hook(self.trainer)
            ret = func(*args, **kwargs)
            if self.with_hook:
                for handle in __f_handles + __b_handles:
                    handle.remove()
            return ret

        return wrapper

    def __deal_with_hook(self, trainer):
        # __last_forward_output, __last_backward_data = {}, {}
        forward_handlers = []
        backward_handlers = []

        for m in trainer.module:
            forward_handlers.append(m.register_forward_hook(hook=self.__hook_forward_fn))
            backward_handlers.append(m.register_full_backward_hook(hook=self.__hook_backward_fn))
        return forward_handlers, backward_handlers

    def __hook_forward_fn(self, module, input, output):
        if self.hook_mute:
            try:
                self.__last_forward_output.pop(module)
            except Exception as _:
                pass
        else:
            print(f'{module.__class__.__name__}前传')
            try:
                last_input, last_output = self.__last_forward_output.pop(module)
            except Exception as _:
                pass
            else:
                flag = True
                for li, i in zip(last_input, input):
                    flag = torch.equal(li, i) and flag
                print(f'输入相等: {flag}')
                flag = True
                for lo, o in zip(last_output, output):
                    flag = torch.equal(lo, o) and flag
                print(f'输出相等: {flag}')
                print('-' * 20)
        # 记录模块的梯度
        self.__last_forward_output[module] = input, output
        return output

    def __hook_backward_fn(self, module, grad_input, grad_output):
        if self.hook_mute:
            try:
                last_input, last_output = self.__last_backward_data.pop(module)
            except Exception as _:
                pass
            else:
                for li, i in zip(last_input, grad_input):
                    if li is None or i is None:
                        print(f'{module.__class__.__name__}反向传播中，{li}或{i}出现了None梯度')
                for lo, o in zip(last_output, grad_output):
                    if lo is None or o is None:
                        print(f'{lo}或{o}出现了None梯度')
        else:
            print(f'{module.__class__.__name__}反向传播')
            try:
                last_input, last_output = self.__last_backward_data.pop(module)
            except Exception as _:
                pass
            else:
                flag = True
                for li, i in zip(last_input, grad_input):
                    if li is None or i is None:
                        print(f'{module.__class__.__name__}反向传播中，{li}或{i}出现了None梯度')
                    else:
                        flag = torch.equal(li, i) and flag
                        print(f'输入梯度相等: {flag}')
                flag = True
                for lo, o in zip(last_output, grad_output):
                    if lo is None or o is None:
                        print(f'{lo}或{o}中出现了None梯度')
                    else:
                        flag = torch.equal(lo, o) and flag
                        print(f'输出梯度相等：{flag}')
                print('-' * 20)
        self.__last_backward_data[module] = grad_input, grad_output