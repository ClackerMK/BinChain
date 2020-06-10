import chainer


class BinaryWeightClip(object):
    """
    Hook function for clipping binary weights.


    """
    name = 'BinaryWeightClip'
    call_for_each_param = True
    timing = 'post'

    def __init__(self, low=-1., high=1.):
        self.low = low
        self.high = high

    def __call__(self, rule, param):
        if hasattr(param, 'binary') and param.binary:
            p = param.data
            with chainer.using_device(param.device):
                xp = param.device.xp
                param.data = xp.clip(p, self.low, self.high)
