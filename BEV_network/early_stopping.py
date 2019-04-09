import numpy as np
import torch
import os


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, folder_path, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            path (str): path to folder where tp save the state dict.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.folder_path = folder_path

    def __call__(self, epoch, train_loss, val_loss , model, optimizer):
        """
        Args:
            epoch: current epoch
            train_loss: trining loss for the current epoch
            val_loss: validation loss for the current epoch
            model: the model that is trained
            optimizer: the used optimizer.
        """

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, train_loss, val_loss, model, optimizer)
        elif self.best_score < score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, train_loss, val_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, epoch, train_loss, val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        file_name = 'epoch_' + str(epoch) + '_checkpoint.pt'
        path = os.path.join(self.folder_path, file_name)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),  # I red in a tutorial that this one is needed, as this contains buffers and parameters that are updated as the model trains.
                    'train_loss': train_loss,
                    'val_loss': val_loss}, path)

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        self.val_loss_min = val_loss
