import torch
from torch.utils.data import DataLoader

torch.manual_seed(100)


def data_loader(dataset, batch_size=32, shuffle=False):
    '''Create and return dataloader form given dataset

    Parameters
    ----------
    dataset : MPIIGaze
        the custom dataset derived from MPIIGaze Dataset  
    batch_size : int, optional
        the batch size for the dataloader, by default 32
    shuffle : bool, optional
        whether to shuffle the data or not, by default False

    Returns
    -------
    DataLoader object
        the final dataloader object
    '''
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
