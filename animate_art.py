"""
Animation Support for Abstract Art Generator

This module provides various animation techniques for generating animated art:
- Weight interpolation between networks
- Coordinate transformations (zoom, rotate, pan)
- Wave/pulse effects
- Spiral/vortex effects
- Time-based modulation
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple, Type
import math

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from art_generator import ArtNetwork, initialize_weights_normal


def interpolate_weights(
    network1: nn.Module,
    network2: nn.Module,
    alpha: float
) -> None:
    """
    Interpolate weights between two networks in-place.

    Args:
        network1: First network (will be modified)
        network2: Second network
        alpha: Interpolation factor (0 = network1, 1 = network2)
    """
    with torch.no_grad():
        for p1, p2 in zip(network1.parameters(), network2.parameters()):
            p1.data = (1 - alpha) * p1.data + alpha * p2.data


def create_coordinate_grid_transformed(
    width: int,
    height: int,
    device: torch.device = torch.device('cpu'),
    rotation: float = 0.0,
    scale: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    wave_amplitude: float = 0.0,
    wave_frequency: float = 1.0,
    spiral_strength: float = 0.0,
) -> torch.Tensor:
    """
    Create a transformed coordinate grid for animations.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        device: Torch device
        rotation: Rotation angle in radians
        scale: Scale factor (1.0 = original, >1 = zoom out, <1 = zoom in)
        offset_x: X-axis offset
        offset_y: Y-axis offset
        wave_amplitude: Amplitude of wave distortion
        wave_frequency: Frequency of wave distortion
        spiral_strength: Strength of spiral/vortex effect

    Returns:
        Tensor of shape (width * height, 2) with transformed coordinates
    """
    x = torch.arange(0, width, dtype=torch.float32, device=device)
    y = torch.arange(0, height, dtype=torch.float32, device=device)

    # Create meshgrid and normalize to [-0.5, 0.5]
    grid_x, grid_y = torch.meshgrid(x / height - 0.5, y / width - 0.5, indexing='ij')

    # Apply transformations
    coords_x = grid_x * scale
    coords_y = grid_y * scale

    # Apply rotation
    if rotation != 0.0:
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        coords_x_rot = coords_x * cos_r - coords_y * sin_r
        coords_y_rot = coords_x * sin_r + coords_y * cos_r
        coords_x, coords_y = coords_x_rot, coords_y_rot

    # Apply spiral/vortex effect
    if spiral_strength != 0.0:
        r = torch.sqrt(coords_x ** 2 + coords_y ** 2)
        angle = torch.atan2(coords_y, coords_x)
        angle = angle + spiral_strength * r
        coords_x = r * torch.cos(angle)
        coords_y = r * torch.sin(angle)

    # Apply wave distortion
    if wave_amplitude != 0.0:
        coords_x = coords_x + wave_amplitude * torch.sin(wave_frequency * coords_y * 2 * math.pi)
        coords_y = coords_y + wave_amplitude * torch.cos(wave_frequency * coords_x * 2 * math.pi)

    # Apply offset
    coords_x = coords_x + offset_x
    coords_y = coords_y + offset_y

    # Reshape to (width * height, 2)
    coordinates = torch.stack([coords_x.flatten(), coords_y.flatten()], dim=1)

    return coordinates


def generate_frame(
    network: nn.Module,
    width: int,
    height: int,
    device: torch.device,
    **transform_kwargs
) -> NDArray[np.float32]:
    """
    Generate a single animation frame with transformations.

    Args:
        network: Neural network to use
        width: Frame width
        height: Frame height
        device: Torch device
        **transform_kwargs: Transformation parameters for coordinate grid

    Returns:
        NumPy array of shape (width, height, 3)
    """
    network.eval()

    with torch.no_grad():
        coordinates = create_coordinate_grid_transformed(
            width, height, device, **transform_kwargs
        )
        colors = network(coordinates)

    # Reshape to image dimensions
    image = colors.cpu().numpy().reshape(width, height, 3)

    return image


# Animation generators

def animate_morph(
    width: int = 256,
    height: int = 256,
    num_frames: int = 60,
    device: Optional[torch.device] = None,
    **network_kwargs
) -> Tuple[list[NDArray[np.float32]], dict]:
    """
    Create morph animation by interpolating between two networks.

    Args:
        width: Frame width
        height: Frame height
        num_frames: Number of frames to generate
        device: Torch device
        **network_kwargs: Arguments for ArtNetwork creation

    Returns:
        Tuple of (list of frames, metadata dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create two different networks
    net1 = ArtNetwork(**network_kwargs).to(device)
    net1.apply(initialize_weights_normal)

    net2 = ArtNetwork(**network_kwargs).to(device)
    net2.apply(initialize_weights_normal)

    # Create interpolation network
    net_interp = ArtNetwork(**network_kwargs).to(device)

    frames = []
    for i in range(num_frames):
        # Smooth interpolation (ease in-out)
        t = i / (num_frames - 1)
        alpha = t * t * (3.0 - 2.0 * t)  # Smoothstep

        # Interpolate weights
        net_interp.load_state_dict(net1.state_dict())
        interpolate_weights(net_interp, net2, alpha)

        # Generate frame
        frame = generate_frame(net_interp, width, height, device)
        frames.append(frame)

    metadata = {
        'type': 'morph',
        'num_frames': num_frames,
        'description': 'Morph between two neural networks'
    }

    return frames, metadata


def animate_zoom(
    width: int = 256,
    height: int = 256,
    num_frames: int = 60,
    zoom_range: Tuple[float, float] = (1.0, 0.2),
    device: Optional[torch.device] = None,
    **network_kwargs
) -> Tuple[list[NDArray[np.float32]], dict]:
    """
    Create zoom animation.

    Args:
        width: Frame width
        height: Frame height
        num_frames: Number of frames
        zoom_range: (start_scale, end_scale) - smaller = zoom in
        device: Torch device
        **network_kwargs: Arguments for ArtNetwork

    Returns:
        Tuple of (list of frames, metadata dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network = ArtNetwork(**network_kwargs).to(device)
    network.apply(initialize_weights_normal)

    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        scale = zoom_range[0] + t * (zoom_range[1] - zoom_range[0])

        frame = generate_frame(network, width, height, device, scale=scale)
        frames.append(frame)

    metadata = {
        'type': 'zoom',
        'num_frames': num_frames,
        'zoom_range': zoom_range,
        'description': f'Zoom from {zoom_range[0]}x to {zoom_range[1]}x'
    }

    return frames, metadata


def animate_rotate(
    width: int = 256,
    height: int = 256,
    num_frames: int = 60,
    rotations: float = 1.0,
    device: Optional[torch.device] = None,
    **network_kwargs
) -> Tuple[list[NDArray[np.float32]], dict]:
    """
    Create rotation animation.

    Args:
        width: Frame width
        height: Frame height
        num_frames: Number of frames
        rotations: Number of full rotations (can be fractional)
        device: Torch device
        **network_kwargs: Arguments for ArtNetwork

    Returns:
        Tuple of (list of frames, metadata dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network = ArtNetwork(**network_kwargs).to(device)
    network.apply(initialize_weights_normal)

    frames = []
    for i in range(num_frames):
        angle = (i / num_frames) * rotations * 2 * math.pi

        frame = generate_frame(network, width, height, device, rotation=angle)
        frames.append(frame)

    metadata = {
        'type': 'rotate',
        'num_frames': num_frames,
        'rotations': rotations,
        'description': f'Rotate {rotations} full rotation(s)'
    }

    return frames, metadata


def animate_wave(
    width: int = 256,
    height: int = 256,
    num_frames: int = 60,
    wave_amplitude: float = 0.1,
    wave_frequency: float = 2.0,
    device: Optional[torch.device] = None,
    **network_kwargs
) -> Tuple[list[NDArray[np.float32]], dict]:
    """
    Create wave/pulse animation.

    Args:
        width: Frame width
        height: Frame height
        num_frames: Number of frames
        wave_amplitude: Amplitude of wave effect
        wave_frequency: Spatial frequency of waves
        device: Torch device
        **network_kwargs: Arguments for ArtNetwork

    Returns:
        Tuple of (list of frames, metadata dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network = ArtNetwork(**network_kwargs).to(device)
    network.apply(initialize_weights_normal)

    frames = []
    for i in range(num_frames):
        # Oscillating wave amplitude
        t = i / num_frames
        current_amp = wave_amplitude * math.sin(t * 2 * math.pi)

        frame = generate_frame(
            network, width, height, device,
            wave_amplitude=current_amp,
            wave_frequency=wave_frequency
        )
        frames.append(frame)

    metadata = {
        'type': 'wave',
        'num_frames': num_frames,
        'wave_amplitude': wave_amplitude,
        'wave_frequency': wave_frequency,
        'description': 'Wave distortion animation'
    }

    return frames, metadata


def animate_spiral(
    width: int = 256,
    height: int = 256,
    num_frames: int = 60,
    spiral_strength: float = 10.0,
    device: Optional[torch.device] = None,
    **network_kwargs
) -> Tuple[list[NDArray[np.float32]], dict]:
    """
    Create spiral/vortex animation.

    Args:
        width: Frame width
        height: Frame height
        num_frames: Number of frames
        spiral_strength: Strength of spiral effect
        device: Torch device
        **network_kwargs: Arguments for ArtNetwork

    Returns:
        Tuple of (list of frames, metadata dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network = ArtNetwork(**network_kwargs).to(device)
    network.apply(initialize_weights_normal)

    frames = []
    for i in range(num_frames):
        t = i / num_frames
        current_strength = spiral_strength * t

        frame = generate_frame(
            network, width, height, device,
            spiral_strength=current_strength
        )
        frames.append(frame)

    metadata = {
        'type': 'spiral',
        'num_frames': num_frames,
        'spiral_strength': spiral_strength,
        'description': 'Spiral/vortex animation'
    }

    return frames, metadata


def animate_pan(
    width: int = 256,
    height: int = 256,
    num_frames: int = 60,
    pan_path: str = 'circular',
    pan_radius: float = 0.3,
    device: Optional[torch.device] = None,
    **network_kwargs
) -> Tuple[list[NDArray[np.float32]], dict]:
    """
    Create panning animation.

    Args:
        width: Frame width
        height: Frame height
        num_frames: Number of frames
        pan_path: Path type ('circular', 'horizontal', 'vertical')
        pan_radius: Radius/distance of pan
        device: Torch device
        **network_kwargs: Arguments for ArtNetwork

    Returns:
        Tuple of (list of frames, metadata dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network = ArtNetwork(**network_kwargs).to(device)
    network.apply(initialize_weights_normal)

    frames = []
    for i in range(num_frames):
        t = (i / num_frames) * 2 * math.pi

        if pan_path == 'circular':
            offset_x = pan_radius * math.cos(t)
            offset_y = pan_radius * math.sin(t)
        elif pan_path == 'horizontal':
            offset_x = pan_radius * math.sin(t)
            offset_y = 0.0
        elif pan_path == 'vertical':
            offset_x = 0.0
            offset_y = pan_radius * math.sin(t)
        else:
            offset_x = offset_y = 0.0

        frame = generate_frame(
            network, width, height, device,
            offset_x=offset_x,
            offset_y=offset_y
        )
        frames.append(frame)

    metadata = {
        'type': 'pan',
        'num_frames': num_frames,
        'pan_path': pan_path,
        'pan_radius': pan_radius,
        'description': f'{pan_path} panning animation'
    }

    return frames, metadata


def animate_combo(
    width: int = 256,
    height: int = 256,
    num_frames: int = 120,
    device: Optional[torch.device] = None,
    **network_kwargs
) -> Tuple[list[NDArray[np.float32]], dict]:
    """
    Create combined animation with multiple effects.

    Args:
        width: Frame width
        height: Frame height
        num_frames: Number of frames
        device: Torch device
        **network_kwargs: Arguments for ArtNetwork

    Returns:
        Tuple of (list of frames, metadata dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network = ArtNetwork(**network_kwargs).to(device)
    network.apply(initialize_weights_normal)

    frames = []
    for i in range(num_frames):
        t = i / num_frames

        # Combine rotation, zoom, and wave
        angle = t * 2 * math.pi
        scale = 1.0 + 0.3 * math.sin(t * 4 * math.pi)
        wave_amp = 0.05 * math.sin(t * 6 * math.pi)

        frame = generate_frame(
            network, width, height, device,
            rotation=angle,
            scale=scale,
            wave_amplitude=wave_amp,
            wave_frequency=2.0
        )
        frames.append(frame)

    metadata = {
        'type': 'combo',
        'num_frames': num_frames,
        'description': 'Combined rotation, zoom, and wave effects'
    }

    return frames, metadata


def save_animation(
    frames: list[NDArray[np.float32]],
    output_path: Path | str,
    fps: int = 30,
    loop: bool = True,
) -> Path:
    """
    Save animation frames to a file (GIF or MP4).

    Args:
        frames: List of frame arrays
        output_path: Output file path (.gif or .mp4)
        fps: Frames per second
        loop: Whether to loop the animation (for GIF)

    Returns:
        Path where animation was saved
    """
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "imageio is required for saving animations. "
            "Install with: pip install imageio[ffmpeg]"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert frames to uint8
    frames_uint8 = [(frame * 255).astype(np.uint8) for frame in frames]

    # Save based on file extension
    if output_path.suffix.lower() == '.gif':
        imageio.mimsave(
            output_path,
            frames_uint8,
            fps=fps,
            loop=0 if loop else 1
        )
    elif output_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        imageio.mimsave(
            output_path,
            frames_uint8,
            fps=fps,
            codec='libx264',
            quality=8
        )
    else:
        raise ValueError(f"Unsupported format: {output_path.suffix}. Use .gif or .mp4")

    return output_path


# Animation type registry
ANIMATION_TYPES = {
    'morph': animate_morph,
    'zoom': animate_zoom,
    'rotate': animate_rotate,
    'wave': animate_wave,
    'spiral': animate_spiral,
    'pan': animate_pan,
    'combo': animate_combo,
}


if __name__ == "__main__":
    # Example usage
    print("Generating zoom animation...")
    frames, metadata = animate_zoom(
        width=128,
        height=128,
        num_frames=30,
        num_neurons=32,
        num_layers=9
    )

    output = save_animation(frames, "test_animation.gif", fps=15)
    print(f"Animation saved to: {output}")
    print(f"Metadata: {metadata}")
