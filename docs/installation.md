# Installation

## Prerequisites

Ohmnivore requires Rust. Install the toolchain via [rustup](https://rustup.rs/):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Building

Clone the repository and build in release mode:

```sh
git clone https://github.com/user/ohmnivore.git
cd ohmnivore
cargo build --release
```

The binary is at `./target/release/ohmnivore`.

## GPU Support

Ohmnivore uses [wgpu](https://wgpu.rs/) for GPU acceleration, which runs on Vulkan, Metal, and DX12. The `--gpu` flag enables it at runtime; no special build flags are needed.

### macOS

Metal works out of the box. No additional drivers required.

### Linux

Install Vulkan drivers for your GPU:

- **NVIDIA**: `sudo apt install nvidia-driver-XXX` (or your distro's equivalent)
- **AMD**: Mesa's RADV driver, typically `sudo apt install mesa-vulkan-drivers`
- **Intel**: `sudo apt install mesa-vulkan-drivers`

Verify with:

```sh
vulkaninfo --summary
```

### Windows

Most modern NVIDIA and AMD drivers include Vulkan support. Intel GPUs use DX12. No extra setup is typically needed.

### CPU Fallback

Without a GPU, omit the `--gpu` flag. The CPU solver uses direct LU decomposition and works everywhere.

## Running Tests

```sh
cargo test                                    # all tests
cargo test --test integration_test            # linear circuit tests
cargo test --test diode_integration_test      # diode tests
cargo test --test transistor_integration_test # BJT/MOSFET tests
cargo test --test gpu_integration_test        # GPU tests (requires GPU)
```

To run ngspice regression tests (requires `ngspice` in PATH):

```sh
cargo test --features ngspice-compare
```
