import mindspore
from mindspore import Tensor, Parameter, ParameterTuple, ms_function
from mindspore import nn, ops

import numpy as np


# same
def exists(val):
    return val is not None


# same
def clamp(value, min_value=None, max_value=None):
    # if exists(min_value):
    #     value = ops.maximum(value, min_value)
    #
    # if exists(max_value):
    #     value = ops.minimum(value, max_value)

    if exists(min_value):
        min_value = Tensor(np.array(min_value, shape=value.shape))
        value = ops.Maximum(value, min_value)

    if exists(max_value):
        max_value = Tensor(np.array(max_value, shape=value.shape))
        value = ops.minimum(value, max_value)

    return value


class EMA(nn.Cell):
    """
    Implements exponential moving average shadowing for your model.
    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """
    def __init__(
        self,
        model,
        beta=0.9999,
        update_after_step=100,
        update_every=10,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        ignore_names=None,
    ):
        super().__init__()
        self.beta = beta
        self.online_model = model
        self.online_params = ParameterTuple(model.trainable_params())
        self.ema_params = self.online_params.clone('ema.')
        self.swap_params = self.online_params.clone('swap.', 'zeros')

        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        self.ignore_names = ignore_names

        self.initted = Parameter(Tensor(False), 'initted')
        self.step = Parameter(Tensor(0, mindspore.int32), 'step')

        self.map = ops.HyperMap()

    def copy_params_from_model_to_ema(self):
        success = self.map(ops.assign, self.ema_params, self.online_params)
        return success

    def get_current_decay(self):
        epoch = clamp(self.step - self.update_after_step - 1, min_value=0.)
        value = 1 - (1 + epoch / self.inv_gamma) ** - self.power

        if epoch <= 0.:
            return Tensor(0., mindspore.float32)

        return clamp(value, min_value=self.min_value, max_value=self.beta)

    # @ms_function
    def update(self):
        success = ops.assign_add(self.step, Tensor(1, mindspore.int32))
        if (self.step % self.update_every) != 0:
            return success
        if self.step <= self.update_after_step:
            success = ops.depend(success, self.copy_params_from_model_to_ema())
            return success

        if not self.initted:
            success = ops.depend(success, self.copy_params_from_model_to_ema())
            ops.assign(self.initted, Tensor(True))

        success = ops.depend(success, self.update_moving_average())

        return success

    def update_moving_average(self):
        def moving_average(current_decay, ma_param, current_param):
            difference = ma_param - current_param
            difference = difference * (1.0 - current_decay)
            return ops.assign(ma_param, ma_param - difference)

        current_decay = self.get_current_decay()

        success = self.map(ops.partial(moving_average, current_decay), self.ema_params, self.online_params)
        return success

    # @ms_function
    def synchronize(self):
        # model -> swap
        success = self.map(ops.assign, self.swap_params, self.online_params)
        # ema -> model
        success = ops.depend(success, self.map(ops.assign, self.online_params, self.ema_params))
        return success

    # @ms_function
    def desynchronize(self):
        # swap -> model
        success = self.map(ops.assign, self.online_params, self.swap_params)
        return success