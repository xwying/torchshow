import numpy as np

def isinteger(x):
    """
    Function to check if np array has integer. Work regardless of type.
    e.g. isinteger(np.array([0., 1.5. 1.])) >>> array([ True, False,  True], dtype=bool)
    
    """
    return np.equal(np.mod(x, 1), 0)
        

def within_0_1(x):
    return (x.min() >= 0) and (x.max() <= 1)


def within_0_255(x):
    return (x.min() >= 0) and (x.max() <= 255)


def calculate_grid_layout(N, img_H, img_W, nrow=None, ncol=None):
    """
    Function to calculate grid_layout
    """
    if (nrow != None and ncol == None):
        ncol = int(np.ceil(N / nrow))
    elif (nrow == None and ncol != None):
        nrow = int(np.ceil(N / ncol))
    else:
        N_sqrt = np.sqrt(N)
        if img_H >= img_W:
            nrow = int(np.floor(N_sqrt))
            ncol = int(np.ceil(N/nrow))
        else:
            ncol = int(np.floor(N_sqrt))
            nrow = int(np.ceil(N/ncol))
    return nrow, ncol
    
    
def tensor_to_array(x):
    # Recursively perform tensor to array conversion
    try:
        import torch # If PyTorch is not installed, TorchShow will not handle torch tensors.
    except ImportError:
        pass
    else:
        if isinstance(x, torch.Tensor):
            return x.detach().clone().cpu().numpy()
        
    if isinstance(x, np.ndarray):
        return x.copy()
    elif isinstance(x, list):
        return [tensor_to_array(e) for e in x]
    else:
        raise TypeError('Found unsupported type ', type(x))


def isnotebook():
    # Check if running in ipython/jupyter notebook environment
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


if __name__ == '__main__':
    for N in range(1,100):
        row, col = calculate_grid_layout(N, 10, 10)
        print(N, row, col, row*col)