import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data import RandomSampler, BatchSampler
from tqdm.auto import tqdm
import torch.nn as nn

from ...explainers.fastshap.image_imputers import ImageImputer
from ...explainers.fastshap.utils import UniformSampler, DatasetRepeat, MaskLayer2d, KLDivLoss, DatasetInputOnly

from copy import deepcopy



class BernoulliSampler:

    def __init__(self, num_players):
        self.num_players = num_players

    def sample(self, batch_size):
        rand = torch.rand(batch_size, self.num_players)
        thresh = torch.rand(batch_size, 1)
        S = (rand>0.5).float()

        return S

def validate(surrogate, loss_fn, data_loader):
    with torch.no_grad():
        # Setup.
        device = next(surrogate.surrogate.parameters()).device
        mean_loss = 0
        N = 0

        for x, y, S in data_loader:
            x = x.to(device)
            y = y.to(device)
            S = S.to(device)
            pred = surrogate(x, S)
            loss = loss_fn(pred, y)
            N += len(x)
            mean_loss += len(x) * (loss - mean_loss) / N

    return mean_loss


def generate_labels(dataset, model, batch_size, num_workers):
    with torch.no_grad():
        # Setup.
        preds = []
        device = next(model.parameters()).device
        loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True,
                            num_workers=num_workers)

        for (x,) in loader:
            pred = model(x.to(device)).cpu()
            preds.append(pred)

    return torch.cat(preds)


class Evaluator(ImageImputer):


    def __init__(self, surrogate, width, height, superpixel_size, dataset_name):
        # Initialize for coalition resizing, number of players.
        super().__init__(width, height, superpixel_size)

        # Store surrogate model.
        self.surrogate = surrogate
        self.dataset_name = dataset_name


    def train_original_model(self,
                             train_data,
                             val_data,
                             original_model,
                             batch_size,
                             max_epochs,
                             loss_fn,
                             validation_samples=1,
                             validation_batch_size=None,
                             lr=1e-3,
                             min_lr=1e-5,
                             lr_factor=0.5,
                             lookback=5,
                             training_seed=None,
                             validation_seed=None,
                             num_workers=0,
                             bar=False,
                             verbose=False):

        # Set up train dataset.
        if isinstance(train_data, np.ndarray):
            train_data = torch.tensor(train_data, dtype=torch.float32)

        if isinstance(train_data, torch.Tensor):
            train_set = TensorDataset(train_data)
        elif isinstance(train_data, Dataset):
            train_set = train_data
        else:
            raise ValueError('train_data must be either tensor or a '
                             'PyTorch Dataset')

        # Set up train data loader.
        random_sampler = RandomSampler(
            train_set, replacement=True,
            num_samples=int(np.ceil(len(train_set) / batch_size))*batch_size)
        batch_sampler = BatchSampler(
            random_sampler, batch_size=batch_size, drop_last=True)
        train_loader = DataLoader(train_set, batch_sampler=batch_sampler,
                                  pin_memory=True, num_workers=num_workers)

        # Set up validation dataset.
        sampler = BernoulliSampler(self.num_players) #######################################
        if validation_seed is not None:
            torch.manual_seed(validation_seed)
        S_val = sampler.sample(len(val_data) * validation_samples)
        if validation_batch_size is None:
            validation_batch_size = batch_size

        if isinstance(val_data, np.ndarray):
            val_data = torch.tensor(val_data, dtype=torch.float32)

        if isinstance(val_data, torch.Tensor):
            # Generate validation labels.
            y_val = generate_labels(TensorDataset(val_data), original_model,
                                    validation_batch_size, num_workers)
            y_val_repeat = y_val.repeat(
                validation_samples, *[1 for _ in y_val.shape[1:]])

            # Create dataset.
            val_data_repeat = val_data.repeat(validation_samples, 1, 1, 1)
            val_set = TensorDataset(val_data_repeat, y_val_repeat, S_val)
        elif isinstance(val_data, Dataset):
            # Generate validation labels.
            y_val = generate_labels(val_data, original_model,
                                    validation_batch_size, num_workers)
            y_val_repeat = y_val.repeat(
                validation_samples, *[1 for _ in y_val.shape[1:]])

            # Create dataset.
            val_set = DatasetRepeat(
                [val_data, TensorDataset(y_val_repeat, S_val)])
        else:
            raise ValueError('val_data must be either tuple of tensors or a '
                             'PyTorch Dataset')

        val_loader = DataLoader(val_set, batch_size=validation_batch_size,
                                pin_memory=True, num_workers=num_workers)

        # Setup for training.
        surrogate = self.surrogate
        device = next(surrogate.parameters()).device
        optimizer = optim.Adam(surrogate.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=lr_factor, patience=lookback // 2, min_lr=min_lr,
            verbose=verbose)
        best_loss = validate(self, loss_fn, val_loader).item()
        best_epoch = 0
        best_model = deepcopy(surrogate)
        loss_list = [best_loss]
        if training_seed is not None:
            torch.manual_seed(training_seed)

        for epoch in range(max_epochs):
            # Batch iterable.
            if bar:
                batch_iter = tqdm(train_loader, desc='Training epoch')
            else:
                batch_iter = train_loader

            for (x,) in batch_iter:
                # Prepare data.
                x = x.to(device)

                # Get original model prediction.
                with torch.no_grad():
                    y = original_model(x)

                # Generate subsets.
                S = sampler.sample(batch_size).to(device=device)

                # Make predictions.
                pred = self.__call__(x, S)
                loss = loss_fn(pred, y)

                # Optimizer step.
                loss.backward()
                optimizer.step()
                surrogate.zero_grad()

            # Evaluate validation loss.
            self.surrogate.eval()
            val_loss = validate(self, loss_fn, val_loader).item()
            self.surrogate.train()

            # Print progress.
            if verbose:
                print('----- Epoch = {} -----'.format(epoch + 1))
                print('Val loss = {:.4f}'.format(val_loss))
                print('')
            scheduler.step(val_loss)
            loss_list.append(val_loss)

            # Check if best model.
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(surrogate)
                best_epoch = epoch
                torch.save(self.surrogate, f'models/{self.dataset_name}_evaluator.pt')
                if verbose:
                    print('New best epoch, loss = {:.4f}'.format(val_loss))
                    print('Saved evaluator model to best_evaluator.pt')
                    print('')
            elif epoch - best_epoch == lookback:
                if verbose:
                    print('Stopping early')
                break

        # Clean up.
        for param, best_param in zip(surrogate.parameters(),
                                     best_model.parameters()):
            param.data = best_param.data
        self.loss_list = loss_list
        self.surrogate.eval()

    def __call__(self, x, S):
        S = self.resize(S)
        return self.surrogate((x, S))
