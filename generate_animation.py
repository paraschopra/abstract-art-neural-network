#!/usr/bin/env python3
"""
Command-line interface for generating animated abstract art.

Usage:
    python generate_animation.py --type morph --output my_animation.gif
    python generate_animation.py --type zoom --frames 120 --fps 60 --output zoom.mp4
"""

import argparse
from pathlib import Path
import time

import torch
import torch.nn as nn

from animate_art import ANIMATION_TYPES, save_animation


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate animated abstract art using neural networks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Animation type
    parser.add_argument(
        "-t", "--type",
        type=str,
        default="morph",
        choices=list(ANIMATION_TYPES.keys()),
        help="Type of animation to generate"
    )

    # Image dimensions
    parser.add_argument(
        "-W", "--width",
        type=int,
        default=256,
        help="Width of the animation in pixels"
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=256,
        help="Height of the animation in pixels"
    )

    # Animation parameters
    parser.add_argument(
        "-f", "--frames",
        type=int,
        default=60,
        help="Number of frames to generate"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for output video/GIF"
    )

    # Output options
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file path (.gif or .mp4)"
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Don't loop the animation (for GIF)"
    )

    # Network architecture
    parser.add_argument(
        "-n", "--neurons",
        type=int,
        default=32,
        help="Number of neurons in hidden layers"
    )
    parser.add_argument(
        "-l", "--layers",
        type=int,
        default=9,
        help="Number of hidden layers (minimum 2)"
    )
    parser.add_argument(
        "-a", "--activation",
        type=str,
        default="Tanh",
        choices=["Tanh", "ReLU", "LeakyReLU", "ELU", "GELU", "Sigmoid"],
        help="Activation function to use"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use (auto-detect if not specified)"
    )

    # Animation-specific parameters
    parser.add_argument(
        "--zoom-start",
        type=float,
        default=1.0,
        help="Starting zoom level (for zoom animation)"
    )
    parser.add_argument(
        "--zoom-end",
        type=float,
        default=0.2,
        help="Ending zoom level (for zoom animation)"
    )
    parser.add_argument(
        "--rotations",
        type=float,
        default=1.0,
        help="Number of full rotations (for rotate animation)"
    )
    parser.add_argument(
        "--wave-amplitude",
        type=float,
        default=0.1,
        help="Wave amplitude (for wave animation)"
    )
    parser.add_argument(
        "--wave-frequency",
        type=float,
        default=2.0,
        help="Wave frequency (for wave animation)"
    )
    parser.add_argument(
        "--spiral-strength",
        type=float,
        default=10.0,
        help="Spiral strength (for spiral animation)"
    )
    parser.add_argument(
        "--pan-path",
        type=str,
        default="circular",
        choices=["circular", "horizontal", "vertical"],
        help="Pan path type (for pan animation)"
    )
    parser.add_argument(
        "--pan-radius",
        type=float,
        default=0.3,
        help="Pan radius/distance (for pan animation)"
    )

    return parser.parse_args()


def get_activation(name: str) -> type[nn.Module]:
    """Get activation function class by name."""
    activations = {
        "Tanh": nn.Tanh,
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "ELU": nn.ELU,
        "GELU": nn.GELU,
        "Sigmoid": nn.Sigmoid,
    }
    return activations[name]


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()

    # Validate output format
    output_path = Path(args.output)
    if output_path.suffix.lower() not in ['.gif', '.mp4', '.avi', '.mov']:
        print(f"Error: Output format must be .gif or .mp4 (got {output_path.suffix})")
        return

    # Determine device
    if args.device:
        device = torch.device(args.device)
        print(f"Using {args.device.upper()}")
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA GPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using Apple Metal GPU")
        else:
            device = torch.device('cpu')
            print("Using CPU")

    # Get activation function
    activation = get_activation(args.activation)

    # Network parameters
    network_kwargs = {
        'num_neurons': args.neurons,
        'num_layers': args.layers,
        'activation': activation,
    }

    # Get animation function
    animate_func = ANIMATION_TYPES[args.type]

    # Prepare animation-specific kwargs
    anim_kwargs = {
        'width': args.width,
        'height': args.height,
        'num_frames': args.frames,
        'device': device,
        **network_kwargs
    }

    # Add type-specific parameters
    if args.type == 'zoom':
        anim_kwargs['zoom_range'] = (args.zoom_start, args.zoom_end)
    elif args.type == 'rotate':
        anim_kwargs['rotations'] = args.rotations
    elif args.type == 'wave':
        anim_kwargs['wave_amplitude'] = args.wave_amplitude
        anim_kwargs['wave_frequency'] = args.wave_frequency
    elif args.type == 'spiral':
        anim_kwargs['spiral_strength'] = args.spiral_strength
    elif args.type == 'pan':
        anim_kwargs['pan_path'] = args.pan_path
        anim_kwargs['pan_radius'] = args.pan_radius

    # Generate animation
    print(f"\nGenerating {args.type} animation...")
    print(f"  Frames: {args.frames}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Architecture: {args.layers} layers x {args.neurons} neurons")
    print(f"  Activation: {args.activation}")
    print()

    start_time = time.time()

    frames, metadata = animate_func(**anim_kwargs)

    generation_time = time.time() - start_time
    print(f"✓ Generated {len(frames)} frames in {generation_time:.2f}s")
    print(f"  ({generation_time/len(frames):.3f}s per frame)")

    # Save animation
    print(f"\nSaving animation to {args.output}...")
    save_start = time.time()

    saved_path = save_animation(
        frames,
        args.output,
        fps=args.fps,
        loop=not args.no_loop
    )

    save_time = time.time() - save_start
    print(f"✓ Animation saved in {save_time:.2f}s")

    # Summary
    print(f"\n{'='*50}")
    print(f"Animation Details:")
    print(f"{'='*50}")
    print(f"Type: {metadata['type']}")
    print(f"Description: {metadata['description']}")
    print(f"Frames: {metadata['num_frames']}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"FPS: {args.fps}")
    print(f"Duration: {metadata['num_frames']/args.fps:.2f}s")
    print(f"File: {saved_path}")
    print(f"File size: {saved_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*50}")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f}s")
    print("\nDone! Enjoy your animated art! 🎨✨")


if __name__ == "__main__":
    main()
