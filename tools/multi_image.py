from torch.nn import Module
from typing import Dict
import torch
import numpy as np
import random
seed = 3407

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU

np.random.seed(seed)
random.seed(seed)

class MultiImageInput(Module):
    """
    Flattens the first and second axes of the image tensor for the encoder.
    """

    def __init__(self, **kwargs):
        """
        Argument/s:
            kwargs - keyword arguments.
        """
        super(MultiImageInput, self).__init__()

    def forward(self, images: torch.FloatTensor) -> Dict[str, torch.FloatTensor]:
        """
        Forward propagation.

        Argument/s:
            images - a batch of images.

        Returns
            Dictionary of outputs.
        """

        images_shape = images.size()
        return {"images": images.view(-1, *images_shape[-3:]),
                "images_per_example": images_shape[1]}


class MultiImageOutput(Module):
    """
    Undoes the flattening of MultiImageInput and concatenates the encoder's last hidden state for the images of each
    example along the spatial axis. Assumes that the second last and last axes of the encoder's last hidden state
    represent the spatial position and the encoded representation for a spatial position.
    """

    def __init__(self, **kwargs):
        """
        Argument/s:
            kwargs - keyword arguments.
        """
        super(MultiImageOutput, self).__init__()

    def forward(self, last_hidden_state: torch.FloatTensor,
                      images_per_example: int) -> Dict[str, torch.FloatTensor]:
        """
        Forward propagation.

        Argument/s:
            last_hidden_state - the last hidden state of a network.
            images_per_example - number of images per example.

        Returns
            Dictionary of outputs.
        """
        last_hidden_state_size = last_hidden_state.size()
        reshape_last_hidden_state = last_hidden_state.reshape(
                    last_hidden_state_size[0] // images_per_example,
                    images_per_example,  #
                    last_hidden_state_size[-2],
                    last_hidden_state_size[-1],
                )
        return {"last_hidden_state": reshape_last_hidden_state.reshape(last_hidden_state_size[0] // images_per_example,
                                                                       -1, last_hidden_state_size[-1])}