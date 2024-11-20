import torch
import torch.nn as nn
import itertools as it
from torch import Tensor
from typing import Sequence

from .mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS

POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


import torch
import torch.nn as nn
from torch import Tensor
from typing import Sequence

from .mlp import MLP, ACTIVATIONS

POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.

    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()

    def _make_feature_extractor(self):
        """
        Creates the feature extractor part of the CNN.
        The architecture is:
        [(CONV -> ACT)*P -> POOL]*(N/P)
        """
        in_channels, in_h, in_w = tuple(self.in_size)
        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        for i, out_channels in enumerate(self.channels):
            # Add convolutional layer
            layers.append(nn.Conv2d(in_channels, out_channels, **self.conv_params))

            # Add activation function
            if self.activation_type == "relu":
                layers.append(nn.ReLU(**self.activation_params))
            elif self.activation_type == "lrelu":
                layers.append(nn.LeakyReLU(**self.activation_params))

            # Update input channels for the next layer
            in_channels = out_channels

            # Add pooling after every `pool_every` layers
            if (i + 1) % self.pool_every == 0:
                pooling_layer = POOLINGS[self.pooling_type](**self.pooling_params)
                layers.append(pooling_layer)
        # ========================
        return nn.Sequential(*layers)

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            dummy_input = torch.zeros(1, *self.in_size)  # Dummy input
            features = self.feature_extractor(dummy_input)
            return features.view(features.size(0), -1).size(1)  # Flatten and get size
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self):
        """
        Creates the MLP part of the CNN:
        (FC -> ACT)*M -> FC
        """
        # TODO:
        #  - Create the MLP part of the model: (FC -> ACT)*M -> Linear
        #  - Use the the MLP implementation from Part 1.
        #  - The first Linear layer should have an input dim of equal to the number of
        #    convolutional features extracted by the convolutional layers.
        #  - The last Linear layer should have an output dim of out_classes.
        mlp: MLP = None
        # ====== YOUR CODE: ======
        dims = [self._n_features()] + list(self.hidden_dims) + [self.out_classes]
        layers = []

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            # Add linear (fully connected) layer
            layers.append(nn.Linear(in_dim, out_dim))

            # Add activation for all layers except the last one
            if out_dim != self.out_classes:
                if self.activation_type == "relu":
                    layers.append(nn.ReLU(**self.activation_params))
                elif self.activation_type == "lrelu":
                    layers.append(nn.LeakyReLU(**self.activation_params))
        mlp = nn.Sequential(*layers)
        # ========================
        return mlp

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        out: Tensor = None
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)  # Extract features
        flattened = features.view(features.size(0), -1)  # Flatten features
        out = self.mlp(flattened)  # Pass through MLP
        # ========================
        return out


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(k % 2 == 1 for k in kernel_sizes)

        # Define activation function
        activation_fn = ACTIVATIONS[activation_type](**activation_params)

        # Create main path
        layers = []
        for i in range(len(channels)):
            conv_in_channels = in_channels if i == 0 else channels[i - 1]
            conv_out_channels = channels[i]
            kernel_size = kernel_sizes[i]

            layers.append(
                nn.Conv2d(
                    in_channels=conv_in_channels,
                    out_channels=conv_out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,  # Preserve spatial dimensions
                    bias=True,
                )
            )

            if batchnorm:
                layers.append(nn.BatchNorm2d(conv_out_channels, eps=1e-05, momentum=0.1))

            layers.append(activation_fn)

            if dropout > 0.0:
                layers.append(nn.Dropout2d(p=dropout))

        # Add final convolution to align output channels
        layers.append(
            nn.Conv2d(
                in_channels=channels[-1],
                out_channels=channels[-1],
                kernel_size=kernel_sizes[-1],
                stride=1,
                padding=kernel_sizes[-1] // 2,
                bias=True,
            )
        )
        self.main_path = nn.Sequential(*layers)

        # Create shortcut path
        if in_channels != channels[-1]:
            self.shortcut_path = nn.Conv2d(
                in_channels=in_channels,
                out_channels=channels[-1],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,  # No bias for shortcuts
            )
        else:
            self.shortcut_path = nn.Identity()

    def forward(self, x: Tensor):
        main_path_out = self.main_path(x)
        shortcut_out = self.shortcut_path(x)
        print(f"Main Path Output Mean: {main_path_out.mean().item()}, Std: {main_path_out.std().item()}")
        print(f"Shortcut Path Output Mean: {shortcut_out.mean().item()}, Std: {shortcut_out.std().item()}")
        out = main_path_out + shortcut_out
        print(f"Combined Output Mean (before ReLU): {out.mean().item()}, Std: {out.std().item()}")
        out = torch.relu(out)
        print(f"Final Output Mean (after ReLU): {out.mean().item()}, Std: {out.std().item()}")
        return out

class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. NOT the outer projections)
            The length determines the number of convolutions, EXCLUDING the
            block input and output convolutions.
            For example, if in_out_channels=10 and inner_channels=[5],
            the block will have three convolutions, with channels 10->5->5->10.
            The first and last arrows are the 1X1 projection convolutions, 
            and the middle one is the inner convolution (corresponding to the kernel size listed in "inner kernel sizes").
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        assert len(inner_channels) > 0
        assert len(inner_channels) == len(inner_kernel_sizes)

        # TODO:
        #  Initialize the base class in the right way to produce the bottleneck block
        #  architecture.
        # ====== YOUR CODE: ======
        # Define the complete channel sequence
        channels = [inner_channels[0]] + inner_channels + [in_out_channels]

        # Define kernel sizes
        kernel_sizes = [1] + inner_kernel_sizes + [1]

        # Initialize the base ResidualBlock
        super().__init__(in_channels=in_out_channels, channels=channels, kernel_sizes=kernel_sizes, **kwargs)
        # ========================


class ResNet(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions (make sure to use the right stride and padding).
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        #  - Use bottleneck blocks if requested and if the number of input and output
        #    channels match for each group of P convolutions.
        #    Reminder: the number of convolutions performed in the bottleneck block is:
        #    2 + len(inner_channels). [1 for each 1X1 proection convolution] + [# inner convolutions].
        # - Use batchnorm and dropout as requested.
        # ====== YOUR CODE: ======
        num_blocks = len(self.channels)  # Total number of convolutions
        for i in range(0, num_blocks, self.pool_every):
            # Channels for this block
            block_channels = self.channels[i : i + self.pool_every]
            
            # Use bottleneck if requested and feasible
            if self.bottleneck and len(block_channels) >= 3:
                layers.append(
                    ResidualBottleneckBlock(
                        in_out_channels=in_channels,
                        inner_channels=block_channels[1:-1],
                        inner_kernel_sizes=[3] * (len(block_channels) - 2),
                        batchnorm=self.batchnorm,
                        dropout=self.dropout,
                        activation_type=self.activation_type,
                        activation_params=self.activation_params,
                    )
                )
                in_channels = block_channels[-1]  # Update in_channels
            else:
                # Use standard ResidualBlock
                layers.append(
                    ResidualBlock(
                        in_channels=in_channels,
                        channels=block_channels,
                        kernel_sizes=[3] * len(block_channels),
                        batchnorm=self.batchnorm,
                        dropout=self.dropout,
                        activation_type=self.activation_type,
                        activation_params=self.activation_params,
                    )
                )
                in_channels = block_channels[-1]  # Update in_channels
            
            # Add pooling layer after P convolutions (except for the last group)
            if (i + self.pool_every) < num_blocks:
                pooling_layer = POOLINGS[self.pooling_type](**self.pooling_params)
                layers.append(pooling_layer)
        # ========================
        seq = nn.Sequential(*layers)
        return seq

