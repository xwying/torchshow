import numpy as np
import torch


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
    if isinstance(x, torch.Tensor):
        return x.detach().clone().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x.copy()
    elif isinstance(x, list):
        return [tensor_to_array(e) for e in x]
    else:
        raise TypeError('Found unsupported type')




if __name__ == '__main__':
    for N in range(1,100):
        row, col = calculate_grid_layout(N, 10, 10)
        print(N, row, col, row*col)