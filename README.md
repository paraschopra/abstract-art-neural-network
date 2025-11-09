# Abstract Art Neural Network Generator

Generate beautiful abstract art using neural networks! This project uses deep neural networks to map 2D coordinates to RGB colors, creating unique and mesmerizing artistic patterns.

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## Example Art

![art 1](https://cdn-images-1.medium.com/max/800/1*lwNsGQZpGf-m6vUmku68CQ.png)

![art 2](https://cdn-images-1.medium.com/max/800/1*HWvXPk8GU35sxJGcimdk5w.png)

![art 3](https://cdn-images-1.medium.com/max/800/1*f8j5FgSTjpImJqVt5JdNQA.png)

## About

This project demonstrates how neural networks can be used as creative tools. By training a network to map coordinates to colors, we get infinite varieties of abstract art. Each network creates a unique piece, and you can control the style by adjusting architecture parameters like depth, width, and activation functions.

**Tutorial**: [Making deep neural networks paint to understand how they work](https://towardsdatascience.com/making-deep-neural-networks-paint-to-understand-how-they-work-4be0901582ee)

## What's New (2025 Modernization)

This codebase has been modernized with:

- **Python Module** (`art_generator.py`): Clean, documented, type-annotated code
- **Command-Line Interface** (`generate_art.py`): Generate art from the terminal
- **Animation Support** (`animate_art.py`, `generate_animation.py`): Create mesmerizing animated art! ✨
- **Modern PyTorch Practices**: GPU support (CUDA/MPS), efficient tensor operations
- **Better Project Structure**: Proper requirements, .gitignore, modular design
- **Enhanced Jupyter Notebook**: Uses the module, better documentation
- **Type Hints**: Full type annotations for better IDE support
- **Improved Documentation**: Comprehensive docstrings and examples

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/paraschopra/abstract-art-neural-network.git
cd abstract-art-neural-network
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

That's it! You're ready to generate art.

## Usage

### Command Line Interface (Quickest)

Generate art with a single command:

```bash
# Basic usage - generates a 128x128 image
python generate_art.py

# High resolution with custom output name
python generate_art.py --width 1920 --height 1080 --output my_art.png

# Experiment with architecture
python generate_art.py --neurons 64 --layers 12 --activation ReLU

# Generate multiple images at once
python generate_art.py --batch 10

# Full control
python generate_art.py \
    --width 1024 \
    --height 768 \
    --neurons 32 \
    --layers 15 \
    --activation Tanh \
    --output masterpiece.png \
    --show
```

#### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-W, --width` | Image width in pixels | 128 |
| `-H, --height` | Image height in pixels | 128 |
| `-o, --output` | Output filename | Random |
| `-n, --neurons` | Neurons per hidden layer | 16 |
| `-l, --layers` | Number of hidden layers | 9 |
| `-a, --activation` | Activation function (Tanh, ReLU, LeakyReLU, ELU, GELU, Sigmoid) | Tanh |
| `-b, --batch` | Number of images to generate | 1 |
| `--device` | Device to use (cpu, cuda, mps) | Auto-detect |
| `--show` | Display the image | False |
| `--no-save` | Don't save to disk | False |

### Python API

Use the module in your own Python scripts:

```python
from art_generator import create_and_generate
import torch.nn as nn

# Generate art
network, image = create_and_generate(
    width=512,
    height=512,
    num_neurons=32,
    num_layers=10,
    activation=nn.Tanh,
    save=True,
    output_path="my_art.png"
)

# The network can be reused at different resolutions
from art_generator import generate_image
high_res = generate_image(network, width=1920, height=1080)
```

### Animations (NEW!)

Generate mesmerizing animated art with multiple effects:

```bash
# Morph between two networks
python generate_animation.py --type morph --output morph.gif

# Zoom animation
python generate_animation.py --type zoom --output zoom.mp4 --fps 60

# Rotation animation
python generate_animation.py --type rotate --rotations 2 --output spin.gif

# Wave/pulse effect
python generate_animation.py --type wave --output wave.mp4

# Spiral/vortex effect
python generate_animation.py --type spiral --output vortex.gif

# Panning animation (circular, horizontal, or vertical)
python generate_animation.py --type pan --pan-path circular --output pan.mp4

# Combined effects (rotation + zoom + wave)
python generate_animation.py --type combo --frames 120 --output combo.gif

# High-res animation with custom parameters
python generate_animation.py \
    --type rotate \
    --width 512 \
    --height 512 \
    --frames 120 \
    --fps 60 \
    --neurons 64 \
    --layers 12 \
    --output masterpiece.mp4
```

#### Animation Options

| Option | Description | Default |
|--------|-------------|---------|
| `-t, --type` | Animation type (morph, zoom, rotate, wave, spiral, pan, combo) | morph |
| `-W, --width` | Animation width | 256 |
| `-H, --height` | Animation height | 256 |
| `-f, --frames` | Number of frames | 60 |
| `--fps` | Frames per second | 30 |
| `-o, --output` | Output file (.gif or .mp4) | Required |
| `--zoom-start/end` | Zoom range | 1.0 / 0.2 |
| `--rotations` | Full rotations | 1.0 |
| `--wave-amplitude` | Wave strength | 0.1 |
| `--spiral-strength` | Spiral strength | 10.0 |
| `--pan-path` | Pan type (circular, horizontal, vertical) | circular |

### Jupyter Notebook (Interactive)

For interactive exploration and experimentation:

```bash
jupyter notebook generate-art.ipynb
```

The notebook includes:
- Quick start guide
- Experiments with network depth and width
- Comparison of different activation functions
- High-resolution generation examples

## How It Works

The system uses a simple but powerful idea:

1. **Neural Network**: A feedforward network takes (x, y) coordinates as input
2. **Coordinate Mapping**: Coordinates are normalized and fed through hidden layers
3. **Color Output**: The network outputs RGB values for each pixel
4. **Random Initialization**: Each network starts with random weights, creating unique art

### Key Features

- **Resolution Independent**: Generate at any size from the same network
- **Fast Generation**: GPU-accelerated when available (CUDA or Apple Metal)
- **Highly Customizable**: Control depth, width, and activation functions
- **Deterministic**: Same network always produces the same art
- **No Training Required**: Uses random initialization creatively

## Project Structure

```
abstract-art-neural-network/
├── art_generator.py          # Core module with modern Python code
├── generate_art.py           # Command-line interface for static art
├── animate_art.py            # Animation module with multiple effects
├── generate_animation.py     # Command-line interface for animations
├── generate-art.ipynb        # Interactive Jupyter notebook
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
├── README.md                # This file
├── IMPROVEMENTS.md          # Future enhancement ideas
└── LICENSE                  # MIT License
```

## Tips for Great Art

1. **Experiment with activations**: ReLU creates sharp edges, Tanh creates smooth gradients
2. **Deeper networks**: More layers (15-30) create more complex patterns
3. **Wider networks**: More neurons (64-128) add detail and variation
4. **Batch generation**: Generate many images and pick your favorites
5. **High resolution**: Networks are resolution-independent - render at any size!

## Performance

- **CPU**: ~1 second for 128x128, ~10 seconds for 1080p
- **GPU (CUDA)**: ~0.1 seconds for 128x128, ~1 second for 1080p
- **Memory**: Minimal - even large networks use <100MB

## Contributing

Contributions are welcome! Check out [IMPROVEMENTS.md](IMPROVEMENTS.md) for ideas on what to work on.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/paraschopra/abstract-art-neural-network.git
cd abstract-art-neural-network
pip install -e .
```

## Future Enhancements

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for a comprehensive list of potential improvements including:

- Animation support
- Web interface
- Style control and conditioning
- 3D art generation
- And much more!

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- NumPy ≥ 1.24
- Matplotlib ≥ 3.7
- Pillow ≥ 10.0

For interactive notebook:
- Jupyter ≥ 1.0
- IPython ≥ 8.12

## License

MIT License - see LICENSE file for details.

## Citation

If you use this in your research or project, please cite:

```bibtex
@misc{chopra2025abstractart,
  author = {Paras Chopra},
  title = {Abstract Art Neural Network Generator},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/paraschopra/abstract-art-neural-network}
}
```

## Author

Made by [@paraschopra](https://twitter.com/paraschopra)

## Acknowledgments

- Tutorial article on [Towards Data Science](https://towardsdatascience.com/making-deep-neural-networks-paint-to-understand-how-they-work-4be0901582ee)
- Thanks to Divyanshu Kalra for optimization suggestions
- Thanks to [@AlexisVLRT](https://github.com/AlexisVLRT) for contributions

---

**Enjoy creating abstract art! Share your creations with #NeuralArt** 🎨
