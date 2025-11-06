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
        input_features: Number of input features (2 for basic coordinates,
                       more if using Fourier features) (default: 2)
    """

    def __init__(
        self,
        activation: Type[nn.Module] = nn.Tanh,
        num_neurons: int = 16,
        num_layers: int = 9,
        input_features: int = 2,
    ) -> None:
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        super().__init__()

        # Input layer: coordinates (2D or with Fourier features) -> hidden layer
        layers: list[nn.Module] = [
            nn.Linear(input_features, num_neurons, bias=True),
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
    device: torch.device = torch.device('cpu'),
    scale: float = 1.0,
    fourier_features: bool = False,
    num_frequencies: int = 5,
) -> torch.Tensor:
    """
    Create a normalized coordinate grid for the image.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        device: Torch device to use (CPU or CUDA)
        scale: Coordinate space scale (lower = zoomed in/finer texture, higher = zoomed out)
        fourier_features: Whether to add Fourier feature encoding for texture control
        num_frequencies: Number of frequency components to add (if fourier_features=True)

    Returns:
        Tensor of shape (width * height, 2) or (width * height, 2 + 2*num_frequencies)
        with normalized coordinates
    """
    x = torch.arange(0, width, dtype=torch.float32, device=device)
    y = torch.arange(0, height, dtype=torch.float32, device=device)

    # Create meshgrid and normalize to [-0.5, 0.5], then scale
    grid_x, grid_y = torch.meshgrid(
        (x / height - 0.5) * scale,
        (y / width - 0.5) * scale,
        indexing='ij'
    )

    # Reshape to (width * height, 2)
    coordinates = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    # Add Fourier features for texture control
    if fourier_features and num_frequencies > 0:
        fourier_coords = []
        for freq in range(1, num_frequencies + 1):
            # Add sin and cos at different frequencies
            freq_scale = 2.0 ** freq * np.pi
            fourier_coords.append(torch.sin(coordinates * freq_scale))
            fourier_coords.append(torch.cos(coordinates * freq_scale))

        # Concatenate original coordinates with Fourier features
        coordinates = torch.cat([coordinates] + fourier_coords, dim=1)

    return coordinates


def generate_image(
    network: nn.Module,
    width: int = 128,
    height: int = 128,
    device: Optional[torch.device] = None,
    scale: float = 1.0,
    fourier_features: bool = False,
    num_frequencies: int = 5,
) -> NDArray[np.float32]:
    """
    Generate an image using the neural network.

    Args:
        network: Neural network to use for generation
        width: Image width in pixels
        height: Image height in pixels
        device: Torch device (defaults to network's device)
        scale: Coordinate space scale for texture control
        fourier_features: Whether to use Fourier feature encoding
        num_frequencies: Number of frequency components

    Returns:
        NumPy array of shape (width, height, 3) with RGB values in [0, 1]
    """
    if device is None:
        device = next(network.parameters()).device

    network.eval()

    with torch.no_grad():
        coordinates = create_coordinate_grid(
            width, height, device,
            scale=scale,
            fourier_features=fourier_features,
            num_frequencies=num_frequencies
        )
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


def get_color_palette(palette_name: str) -> NDArray[np.float32]:
    """
    Get a predefined color palette.

    Args:
        palette_name: Name of the palette (grayscale, warm, cool, vibrant, pastel, earth)

    Returns:
        NumPy array of RGB colors, shape (num_colors, 3)
    """
    palettes = {
        'grayscale': np.array([
            [0.0, 0.0, 0.0],  # Black
            [0.33, 0.33, 0.33],  # Dark gray
            [0.67, 0.67, 0.67],  # Light gray
            [1.0, 1.0, 1.0],  # White
        ]),
        'warm': np.array([
            [0.8, 0.1, 0.1],  # Red
            [0.9, 0.4, 0.1],  # Orange
            [0.95, 0.7, 0.2],  # Yellow
            [0.9, 0.8, 0.6],  # Cream
        ]),
        'cool': np.array([
            [0.1, 0.2, 0.4],  # Dark blue
            [0.2, 0.4, 0.7],  # Blue
            [0.3, 0.7, 0.8],  # Cyan
            [0.7, 0.9, 0.9],  # Light cyan
        ]),
        'vibrant': np.array([
            [1.0, 0.0, 0.5],  # Hot pink
            [1.0, 0.5, 0.0],  # Orange
            [0.0, 1.0, 0.5],  # Spring green
            [0.0, 0.5, 1.0],  # Azure
            [0.5, 0.0, 1.0],  # Purple
        ]),
        'pastel': np.array([
            [1.0, 0.8, 0.85],  # Pink
            [0.8, 0.9, 1.0],  # Light blue
            [0.9, 1.0, 0.8],  # Light green
            [1.0, 0.95, 0.8],  # Cream
            [0.95, 0.85, 1.0],  # Lavender
        ]),
        'earth': np.array([
            [0.3, 0.2, 0.1],  # Dark brown
            [0.5, 0.35, 0.2],  # Brown
            [0.7, 0.5, 0.3],  # Tan
            [0.4, 0.5, 0.3],  # Olive
            [0.8, 0.7, 0.5],  # Sand
        ]),
    }

    if palette_name not in palettes:
        available = ', '.join(palettes.keys())
        raise ValueError(f"Unknown palette '{palette_name}'. Available: {available}")

    return palettes[palette_name]


def apply_color_palette(
    image: NDArray[np.float32],
    palette: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Apply a color palette to an image by mapping each pixel to the nearest palette color.

    Args:
        image: RGB image array of shape (width, height, 3) with values in [0, 1]
        palette: Array of RGB colors, shape (num_colors, 3) with values in [0, 1]

    Returns:
        Image with colors constrained to the palette
    """
    width, height, _ = image.shape
    pixels = image.reshape(-1, 3)  # Flatten to (width*height, 3)

    # Compute distances to each palette color (vectorized)
    # distances shape: (width*height, num_colors)
    distances = np.sqrt(((pixels[:, np.newaxis, :] - palette[np.newaxis, :, :]) ** 2).sum(axis=2))

    # Find nearest palette color for each pixel
    nearest_indices = np.argmin(distances, axis=1)

    # Map to palette colors
    quantized_pixels = palette[nearest_indices]

    # Reshape back to image
    return quantized_pixels.reshape(width, height, 3)


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
    seed: Optional[int] = None,
    scale: float = 1.0,
    fourier_features: bool = False,
    num_frequencies: int = 5,
    palette: Optional[str] = None,
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
        seed: Random seed for reproducible generation (None for random)
        scale: Coordinate space scale for texture control (lower = finer details)
        fourier_features: Enable Fourier feature encoding for richer textures
        num_frequencies: Number of Fourier frequency components (if fourier_features=True)
        palette: Color palette name (None, 'grayscale', 'warm', 'cool', 'vibrant', 'pastel', 'earth')
        **network_kwargs: Additional arguments for ArtNetwork

    Returns:
        Tuple of (network, image_array)

    Example:
        >>> net, img = create_and_generate(
        ...     width=256,
        ...     height=256,
        ...     num_neurons=32,
        ...     num_layers=8,
        ...     activation=nn.ReLU,
        ...     seed=42,
        ...     scale=2.0,
        ...     palette='warm'
        ... )
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Calculate input features based on Fourier encoding
    input_features = 2  # Base (x, y) coordinates
    if fourier_features and num_frequencies > 0:
        input_features = 2 + 2 * num_frequencies * 2  # Original + sin/cos for each freq

    # Create and initialize network
    network = ArtNetwork(input_features=input_features, **network_kwargs).to(device)
    network.apply(initialize_weights_normal)

    # Generate image
    image = generate_image(
        network, width, height, device,
        scale=scale,
        fourier_features=fourier_features,
        num_frequencies=num_frequencies
    )

    # Apply color palette if specified
    if palette is not None:
        palette_colors = get_color_palette(palette)
        image = apply_color_palette(image, palette_colors)

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
