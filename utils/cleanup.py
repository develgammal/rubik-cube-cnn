import torch
import gc
import matplotlib.pyplot as plt

# clean up memory
def cleanup(*args):
    """
    Clean up memory by closing plots, clearing GPU cache, deleting objects passed as arguments such as loaders and models, and forcing garbage collection.

    Args:
        *args: Any number of objects to be deleted for freeing up memory.
    """
    # Close all plots
    plt.close('all')

    # Clear GPU cache if applicable
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'mps'):
        # Checks if the system has MPS (Metal Performance Shaders) support, which is relevant for Mac devices
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Delete each object passed as argument typically loaders and models
    for obj in args:
        del obj

    # Force garbage collection to free memory
    gc.collect()
