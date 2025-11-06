"""
Abstract Art Generator using Neural Networks

This module provides tools to generate abstract art using neural networks.
The networks map 2D coordinates to RGB colors, creating unique artistic patterns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from numpy.typing import NDArray


class ArtNetwork(nn.Module):
    """
    A neural network that maps 2D coordinates to RGB colors.

    The network takes normalized (x, y) coordinates as input and outputs
    RGB values in the range [0, 1].

    Args:
        activation: PyTorch activation function class (default: nn.Tanh)
        num_neurons: Number of neurons in hidden layers (default: 16)
        num_layers: Number of hidden layers (minimum 2) (default: 9)
    """

    def __init__(
        self,
        activation: Type[nn.Module] = nn.Tanh,
        num_neurons: int = 16,
        num_layers: int = 9,
    ) -> None:
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        super().__init__()

        # Input layer: 2D coordinates -> hidden layer
        layers: list[nn.Module] = [
            nn.Linear(2, num_neurons, bias=True),
            activation()
        ]

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(num_neurons, num_neurons, bias=False),
                activation()
            ])

        # Output layer: hidden -> RGB (3 channels)
        layers.extend([
            nn.Linear(num_neurons, 3, bias=False),
            nn.Sigmoid()  # Ensure output is in [0, 1]
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.layers(x)


def initialize_weights_normal(model: nn.Module) -> None:
    """
    Initialize linear layer weights with normal distribution.

    Args:
        model: PyTorch model to initialize
    """
    if isinstance(model, nn.Linear):
        nn.init.normal_(model.weight)


def create_coordinate_grid(
    width: int,
    height: int,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Create a normalized coordinate grid for the image.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        device: Torch device to use (CPU or CUDA)

    Returns:
        Tensor of shape (width * height, 2) with normalized coordinates
    """
    x = torch.arange(0, width, dtype=torch.float32, device=device)
    y = torch.arange(0, height, dtype=torch.float32, device=device)

    # Create meshgrid and normalize to [-0.5, 0.5]
    grid_x, grid_y = torch.meshgrid(x / height - 0.5, y / width - 0.5, indexing='ij')

    # Reshape to (width * height, 2)
    coordinates = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    return coordinates


def generate_image(
    network: nn.Module,
    width: int = 128,
    height: int = 128,
    device: Optional[torch.device] = None,
) -> NDArray[np.float32]:
    """
    Generate an image using the neural network.

    Args:
        network: Neural network to use for generation
        width: Image width in pixels
        height: Image height in pixels
        device: Torch device (defaults to network's device)

    Returns:
        NumPy array of shape (width, height, 3) with RGB values in [0, 1]
    """
    if device is None:
        device = next(network.parameters()).device

    network.eval()

    with torch.no_grad():
        coordinates = create_coordinate_grid(width, height, device)
        colors = network(coordinates)

    # Reshape to image dimensions
    image = colors.cpu().numpy().reshape(width, height, 3)

    return image


def plot_image(
    image: NDArray[np.float32],
    fig_size: int = 4,
    show: bool = True
) -> plt.Figure:
    """
    Display the generated image using matplotlib.

    Args:
        image: RGB image array of shape (width, height, 3)
        fig_size: Figure size in inches
        show: Whether to display the figure

    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(image, interpolation='nearest', vmin=0, vmax=1)
    plt.axis('off')

    if show:
        plt.show()

    return fig


def save_image(
    image: NDArray[np.float32],
    output_path: Optional[Path | str] = None
) -> Path:
    """
    Save the generated image to disk.

    Args:
        image: RGB image array of shape (width, height, 3)
        output_path: Output file path (generates random name if None)

    Returns:
        Path where the image was saved
    """
    if output_path is None:
        output_path = Path(f"{np.random.randint(100000)}.png")
    else:
        output_path = Path(output_path)

    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.imsave(output_path, image)

    return output_path


def create_and_generate(
    width: int = 128,
    height: int = 128,
    save: bool = True,
    output_path: Optional[Path | str] = None,
    show: bool = True,
    fig_size: int = 4,
    device: Optional[torch.device] = None,
    **network_kwargs,
) -> Tuple[ArtNetwork, NDArray[np.float32]]:
    """
    Create a new network and generate an abstract art image.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        save: Whether to save the image to disk
        output_path: Output file path (random name if None)
        show: Whether to display the image
        fig_size: Figure size for display
        device: Torch device (CPU or CUDA)
        **network_kwargs: Additional arguments for ArtNetwork

    Returns:
        Tuple of (network, image_array)

    Example:
        >>> net, img = create_and_generate(
        ...     width=256,
        ...     height=256,
        ...     num_neurons=32,
        ...     num_layers=8,
        ...     activation=nn.ReLU
        ... )
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create and initialize network
    network = ArtNetwork(**network_kwargs).to(device)
    network.apply(initialize_weights_normal)

    # Generate image
    image = generate_image(network, width, height, device)

    # Display image
    if show:
        plot_image(image, fig_size, show=True)

    # Save image
    if save:
        saved_path = save_image(image, output_path)
        print(f"Image saved to: {saved_path}")

    return network, image


if __name__ == "__main__":
    # Example usage
    print("Generating abstract art...")
    network, image = create_and_generate(
        width=128,
        height=128,
        num_neurons=32,
        num_layers=9,
        save=True,
        show=False
    )
    print("Done!")
