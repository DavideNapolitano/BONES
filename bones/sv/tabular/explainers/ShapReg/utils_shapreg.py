import numpy as np
import pickle

def crossentropyloss(pred, target):
    if pred.ndim == 1:
        pred = pred[:, np.newaxis]
        pred = np.concatenate((1 - pred, pred), axis=1)

    if pred.shape == target.shape:
        # Soft cross entropy loss.
        pred = np.clip(pred, a_min=1e-12, a_max=1-1e-12)
        return - np.sum(np.log(pred) * target, axis=1)
    else:
        # Standard cross entropy loss.
        return - np.log(pred[np.arange(len(pred)), target])


def mseloss(pred, target):
    if len(pred.shape) == 1:
        pred = pred[:, np.newaxis]
    if len(target.shape) == 1:
        target = target[:, np.newaxis]
    return np.sum((pred - target) ** 2, axis=1)


class ShapleyValues:
    def __init__(self, values, std):
        self.values = values
        self.std = std

    def save(self, filename):
        if isinstance(filename, str):
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise TypeError('filename must be str')

    def __repr__(self):
        with np.printoptions(precision=2, threshold=12, floatmode='fixed'):
            return 'Shapley Values(\n  (Mean): {}\n  (Std):  {}\n)'.format(self.values, self.std)


def load(filename):
    with open(filename, 'rb') as f:
        shapley_values = pickle.load(f)
        if isinstance(shapley_values, ShapleyValues):
            return shapley_values
        else:
            raise ValueError('object is not instance of ShapleyValues class')


class CooperativeGame:
    def __init__(self):
        raise NotImplementedError

    def __call__(self, S):
        raise NotImplementedError

    def grand(self):
        return self.__call__(np.ones((1, self.players), dtype=bool))[0]

    def null(self):
        return self.__call__(np.zeros((1, self.players), dtype=bool))[0]


class PredictionGame(CooperativeGame):

    def __init__(self, extension, sample, groups=None):
        # Add batch dimension to sample.
        if sample.ndim == 1:
            sample = sample[np.newaxis]
        elif sample.shape[0] != 1:
            raise ValueError('sample must have shape (ndim,) or (1,ndim)')

        self.extension = extension
        self.sample = sample

        # Store feature groups.
        num_features = sample.shape[1]
        if groups is None:
            self.players = num_features
            self.groups_matrix = None
        else:
            # Verify groups.
            inds_list = []
            for group in groups:
                inds_list += list(group)
            assert np.all(np.sort(inds_list) == np.arange(num_features))

            # Map groups to features.
            self.players = len(groups)
            self.groups_matrix = np.zeros(
                (len(groups), num_features), dtype=bool)
            for i, group in enumerate(groups):
                self.groups_matrix[i, group] = True

        # Caching.
        self.sample_repeat = sample

    def __call__(self, S):
        # Try to use caching for repeated data.
        if len(S) != len(self.sample_repeat):
            self.sample_repeat = self.sample.repeat(len(S), 0)
        input_data = self.sample_repeat

        # Apply group transformation.
        if self.groups_matrix is not None:
            S = np.matmul(S, self.groups_matrix)

        # Evaluate.
        return self.extension(input_data, S)


class PredictionLossGame(CooperativeGame):

    def __init__(self, extension, sample, label, loss, groups=None):
        # Add batch dimension to sample.
        if sample.ndim == 1:
            sample = sample[np.newaxis]

        # Add batch dimension to label.
        if np.isscalar(label):
            label = np.array([label])

        # Convert label dtype if necessary.
        if loss is crossentropyloss:
            # Make sure not soft cross entropy.
            if (label.ndim <= 1) or (label.shape[1] == 1):
                # Only convert if float.
                if np.issubdtype(label.dtype, np.floating):
                    label = label.astype(int)

        self.extension = extension
        self.sample = sample
        self.label = label
        self.loss = loss

        # Store feature groups.
        num_features = sample.shape[1]
        if groups is None:
            self.players = num_features
            self.groups_matrix = None
        else:
            # Verify groups.
            inds_list = []
            for group in groups:
                inds_list += list(group)
            assert np.all(np.sort(inds_list) == np.arange(num_features))

            # Map groups to features.
            self.players = len(groups)
            self.groups_matrix = np.zeros(
                (len(groups), num_features), dtype=bool)
            for i, group in enumerate(groups):
                self.groups_matrix[i, group] = True

        # Caching.
        self.sample_repeat = sample
        self.label_repeat = label

    def __call__(self, S):
        # Try to use caching for repeated data.
        if len(S) != len(self.sample_repeat):
            self.sample_repeat = self.sample.repeat(len(S), 0)
            self.label_repeat = self.label.repeat(len(S), 0)
        input_data = self.sample_repeat
        output_label = self.label_repeat

        # Apply group transformation.
        if self.groups_matrix is not None:
            S = np.matmul(S, self.groups_matrix)

        # Evaluate.
        return - self.loss(self.extension(input_data, S), output_label)


class StochasticCooperativeGame:

    def __init__(self):
        raise NotImplementedError

    def __call__(self, S, U):
        raise NotImplementedError

    def sample(self, samples):
        inds = np.random.choice(self.N, size=samples)
        return tuple(arr[inds] for arr in self.exogenous)

    def iterate(self, batch_size):
        ind = 0
        while ind < self.N:
            yield tuple(arr[ind:(ind + batch_size)] for arr in self.exogenous)
            ind += batch_size

    def grand(self, batch_size):
        N = 0
        mean_value = 0
        ones = np.ones((batch_size, self.players), dtype=bool)
        for U in self.iterate(batch_size):
            N += len(U[0])

            # Update mean value.
            value = self.__call__(ones[:len(U[0])], U)
            mean_value += np.sum(value - mean_value, axis=0) / N

        return mean_value

    def null(self, batch_size):
        N = 0
        mean_value = 0
        zeros = np.zeros((batch_size, self.players), dtype=bool)
        for U in self.iterate(batch_size):
            N += len(U[0])

            # Update mean value.
            value = self.__call__(zeros[:len(U[0])], U)
            mean_value += np.sum(value - mean_value, axis=0) / N

        return mean_value


class DatasetLossGame(StochasticCooperativeGame):

    def __init__(self, extension, data, labels, loss, groups=None):
        self.extension = extension
        self.loss = loss
        self.N = len(data)
        assert len(labels) == self.N
        self.exogenous = (data, labels)

        # Store feature groups.
        num_features = data.shape[1]
        if groups is None:
            self.players = num_features
            self.groups_matrix = None
        else:
            # Verify groups.
            inds_list = []
            for group in groups:
                inds_list += list(group)
            assert np.all(np.sort(inds_list) == np.arange(num_features))

            # Map groups to features.
            self.players = len(groups)
            self.groups_matrix = np.zeros(
                (len(groups), num_features), dtype=bool)
            for i, group in enumerate(groups):
                self.groups_matrix[i, group] = True

    def __call__(self, S, U):
        # Unpack exogenous random variable.
        if U is None:
            U = self.sample(len(S))
        x, y = U

        # Possibly convert label datatype.
        if self.loss is crossentropyloss:
            # Make sure not soft cross entropy.
            if (y.ndim == 1) or (y.shape[1] == 1):
                if np.issubdtype(y.dtype, np.floating):
                    y = y.astype(int)

        # Apply group transformation.
        if self.groups_matrix is not None:
            S = np.matmul(S, self.groups_matrix)

        # Evaluate.
        return - self.loss(self.extension(x, S), y)


class DatasetOutputGame(StochasticCooperativeGame):

    def __init__(self, extension, data, loss, groups=None):
        self.extension = extension
        self.loss = loss
        self.N = len(data)
        self.exogenous = (data,)

        # Store feature groups.
        num_features = data.shape[1]
        if groups is None:
            self.players = num_features
            self.groups_matrix = None
        else:
            # Verify groups.
            inds_list = []
            for group in groups:
                inds_list += list(group)
            assert np.all(np.sort(inds_list) == np.arange(num_features))

            # Map groups to features.
            self.players = len(groups)
            self.groups_matrix = np.zeros(
                (len(groups), num_features), dtype=bool)
            for i, group in enumerate(groups):
                self.groups_matrix[i, group] = True

    def __call__(self, S, U):
        # Unpack exogenous random variable.
        if U is None:
            U = self.sample(len(S))
        x = U[0]

        # Apply group transformation.
        if self.groups_matrix is not None:
            S = np.matmul(S, self.groups_matrix)

        # Evaluate.
        return - self.loss(self.extension(x, S),
                           self.extension(x, np.ones(x.shape, dtype=bool)))
    
import numpy as np


class DefaultExtension:
    def __init__(self, values, model):
        self.model = model
        if values.ndim == 1:
            values = values[np.newaxis]
        elif values[0] != 1:
            raise ValueError('values shape must be (dim,) or (1, dim)')
        self.values = values
        self.values_repeat = values

    def __call__(self, x, S):
        # Prepare x.
        if len(x) != len(self.values_repeat):
            self.values_repeat = self.values.repeat(len(x), 0)

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = self.values_repeat[~S]

        # Make predictions.
        return self.model(x_)


class MarginalExtension:
    def __init__(self, data, model):
        self.model = model
        self.data = data
        self.data_repeat = data
        self.samples = len(data)
        # self.x_addr = None
        # self.x_repeat = None

    def __call__(self, x, S):
        # Prepare x and S.
        n = len(x)
        x = x.repeat(self.samples, 0)
        S = S.repeat(self.samples, 0)
        # if self.x_addr != id(x):
        #     self.x_addr = id(x)
        #     self.x_repeat = x.repeat(self.samples, 0)
        # x = self.x_repeat

        # Prepare samples.
        if len(self.data_repeat) != self.samples * n:
            self.data_repeat = np.tile(self.data, (n, 1))

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = self.data_repeat[~S]

        # Make predictions.
        pred = self.model(x_)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)


class UniformExtension:
    def __init__(self, values, categorical_inds, samples, model):
        self.model = model
        self.values = values
        self.categorical_inds = categorical_inds
        self.samples = samples

    def __call__(self, x, S):
        # Prepare x and S.
        n = len(x)
        x = x.repeat(self.samples, 0)
        S = S.repeat(self.samples, 0)

        # Prepare samples.
        samples = np.zeros((n * self.samples, x.shape[1]))
        for i in range(x.shape[1]):
            if i in self.categorical_inds:
                inds = np.random.choice(
                    len(self.values[i]), n * self.samples)
                samples[:, i] = self.values[i][inds]
            else:
                samples[:, i] = np.random.uniform(
                    low=self.values[i][0], high=self.values[i][1],
                    size=n * self.samples)

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = samples[~S]

        # Make predictions.
        pred = self.model(x_)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)


class UniformContinuousExtension:
    def __init__(self, min_vals, max_vals, samples, model):
        self.model = model
        self.min = min_vals
        self.max = max_vals
        self.samples = samples

    def __call__(self, x, S):
        # Prepare x and S.
        x = x.repeat(self.samples, 0)
        S = S.repeat(self.samples, 0)

        # Prepare samples.
        u = np.random.uniform(size=x.shape)
        samples = u * self.min + (1 - u) * self.max

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = samples[~S]

        # Make predictions.
        pred = self.model(x_)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)


class ProductMarginalExtension:
    def __init__(self, data, samples, model):
        self.model = model
        self.data = data
        self.data_repeat = data
        self.samples = samples

    def __call__(self, x, S):
        # Prepare x and S.
        n = len(x)
        x = x.repeat(self.samples, 0)
        S = S.repeat(self.samples, 0)

        # Prepare samples.
        samples = np.zeros((n * self.samples, x.shape[1]))
        for i in range(x.shape[1]):
            inds = np.random.choice(len(self.data), n * self.samples)
            samples[:, i] = self.data[inds, i]

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = samples[~S]

        # Make predictions.
        pred = self.model(x_)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)


class SeparateModelExtension:
    def __init__(self, model_dict):
        self.model_dict = model_dict

    def __call__(self, x, S):
        output = []
        for i in range(len(S)):
            # Extract model.
            row = S[i]
            model = self.model_dict[str(row)]

            # Make prediction.
            output.append(model(x[i:i+1, row]))

        return np.concatenate(output, axis=0)


class ConditionalExtension:
    def __init__(self, conditional_model, samples, model):
        self.model = model
        self.conditional_model = conditional_model
        self.samples = samples
        self.x_addr = None
        self.x_repeat = None

    def __call__(self, x, S):
        # Prepare x.
        if self.x_addr != id(x):
            self.x_addr = id(x)
            self.x_repeat = x.repeat(self.samples, 0)
        x = self.x_repeat

        # Prepare samples.
        S = S.repeat(self.samples, 0)
        samples = self.conditional_model(x, S)

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = samples[~S]

        # Make predictions.
        pred = self.model(x_)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)


class ConditionalSupervisedExtension:
    def __init__(self, surrogate):
        self.surrogate = surrogate

    def __call__(self, x, S):
        return self.surrogate(x, S)