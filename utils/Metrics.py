import torch
import numpy as np

# masked version error functions partly copied from PVCGN
def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


# Mask target value 0 out
class Metrics(object):
    def __init__(self, target, output, mode, null_val=0):
        self.target = target.numpy()
        self.output = output.numpy()
        self.mode = mode
        self.null_val = null_val

    def rmse(self):
        rmse = torch.as_tensor([
            masked_rmse_np(self.output[:, i, :, :], self.target[:, i, :, :], null_val=self.null_val)
            for i in range(self.output.shape[1])
        ])
        if self.mode == 'val':
            rmse = rmse.mean().item()

        return rmse

    def mae(self):
        mae = torch.as_tensor([
            masked_mae_np(self.output[:, i, :, :], self.target[:, i, :, :], null_val=self.null_val)
            for i in range(self.output.shape[1])
        ])
        if self.mode == 'val':
            mae = mae.mean().item()

        return mae

    def mape(self):
        mape = torch.as_tensor([
            masked_mape_np(self.output[:, i, :, :], self.target[:, i, :, :], null_val=self.null_val)
            for i in range(self.output.shape[1])
        ]) * 100
        if self.mode == 'val':
            mape = mape.mean().item()

        return mape

    def all(self):
        rmse = self.rmse()
        mae = self.mae()
        mape = self.mape()

        return rmse, mae, mape
