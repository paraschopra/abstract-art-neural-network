#!/usr/bin/env python3
"""
Command-line interface for generating abstract art using neural networks.

Usage:
    python generate_art.py --width 256 --height 256 --output my_art.png
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from art_generator import create_and_generate


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate abstract art using neural networks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Image dimensions
    parser.add_argument(
        "-W", "--width",
        type=int,
        default=128,
        help="Width of the generated image in pixels"
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=128,
        help="Height of the generated image in pixels"
    )

    # Output options
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (random name if not specified)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the image to disk"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the image (requires GUI)"
    )

    # Network architecture
    parser.add_argument(
        "-n", "--neurons",
        type=int,
        default=16,
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

    # Style control parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Coordinate space scale (lower = finer details, higher = coarser)"
    )
    parser.add_argument(
        "--fourier",
        action="store_true",
        help="Enable Fourier feature encoding for richer textures"
    )
    parser.add_argument(
        "--num-frequencies",
        type=int,
        default=5,
        help="Number of Fourier frequency components (used with --fourier)"
    )
    parser.add_argument(
        "--palette",
        type=str,
        default=None,
        choices=["grayscale", "warm", "cool", "vibrant", "pastel", "earth"],
        help="Apply a color palette constraint"
    )

    # Display options
    parser.add_argument(
        "-s", "--fig-size",
        type=int,
        default=4,
        help="Figure size in inches for display"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use (auto-detect if not specified)"
    )

    # Batch generation
    parser.add_argument(
        "-b", "--batch",
        type=int,
        default=1,
        help="Number of images to generate"
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

    # Determine device
    if args.device:
        device = torch.device(args.device)
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

    # Generate images
    for i in range(args.batch):
        print(f"\nGenerating image {i + 1}/{args.batch}...")

        # Determine output path for batch generation
        output_path = args.output
        if args.batch > 1 and output_path:
            path = Path(output_path)
            output_path = path.parent / f"{path.stem}_{i+1}{path.suffix}"

        # Generate art
        network, image = create_and_generate(
            width=args.width,
            height=args.height,
            save=not args.no_save,
            output_path=output_path,
            show=args.show,
            fig_size=args.fig_size,
            device=device,
            num_neurons=args.neurons,
            num_layers=args.layers,
            activation=activation,
            seed=args.seed,
            scale=args.scale,
            fourier_features=args.fourier,
            num_frequencies=args.num_frequencies,
            palette=args.palette,
        )

        print(f"✓ Image {i + 1} generated successfully")
        print(f"  Architecture: {args.layers} layers × {args.neurons} neurons")
        print(f"  Activation: {args.activation}")
        print(f"  Dimensions: {args.width}×{args.height}")
        if args.seed is not None:
            print(f"  Seed: {args.seed}")
        if args.scale != 1.0:
            print(f"  Scale: {args.scale}")
        if args.fourier:
            print(f"  Fourier features: {args.num_frequencies} frequencies")
        if args.palette:
            print(f"  Palette: {args.palette}")

    print(f"\nAll done! Generated {args.batch} image(s)")


if __name__ == "__main__":
    main()
