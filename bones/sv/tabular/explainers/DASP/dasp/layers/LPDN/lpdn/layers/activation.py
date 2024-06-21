import numpy as np
import tensorflow as tf
from tensorflow import distributions as dist
from keras.layers import Activation

exp = np.exp
normal = dist.Normal(loc=0., scale=1.)


def _filter_linear(m, v):
    return m, v


def _filter_relu(m, v):
    v = tf.maximum(v, 0.0001)
    s = v**0.5
    m_out = m*normal.cdf(m/s) + s*normal.prob(m/s)
    v_out = (m**2 + v)*normal.cdf(m/s) + (m*s)*normal.prob(m/s) - m_out**2
    return m_out, v_out

# def _filter_softmax(m, v):
#     v = tf.maximum(v, 0.0001)
#     s = v**0.5
#     m_out = exp(m + v/2)
#     v_out = (exp(v) - 1) * exp(2*m + v)
#     return m_out, v_out


ACTIVATIONS = {
    'linear' : _filter_linear,
    'relu': _filter_relu,
    # 'softmax': _filter_softmax,
}


def filter_activation(activation_name, m, v):
    activation_name = activation_name.lower()
    if activation_name in ACTIVATIONS:
        return ACTIVATIONS[activation_name](m, v)
    else:
        raise Exception("Activation '%s' not supported" % activation_name)


class LPActivation(Activation):
    def __init__(self, activation, **kwargs):
        self.activation_name = activation if isinstance(activation, str) else activation.__name__
        if self.activation_name not in ACTIVATIONS:
            raise Exception("Activation '%s' not supported" % self.activation_name)
        super(LPActivation, self).__init__(activation, **kwargs)

    def call(self, inputs):
        m = inputs[..., 0]
        v = inputs[..., 1]
        m, v = filter_activation(self.activation_name, m, v)
        return tf.stack([m, v], -1)
