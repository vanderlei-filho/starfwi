#!/usr/bin/env python3
"""
Visualization tools for StarFWI simulation snapshots.

Reads binary snapshot files and creates visualizations:
- 2D slices of velocity models
- Wavefield animations
- Comparison plots for FWI iterations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import struct
from typing import Tuple, Optional
from enum import IntEnum


class FieldType(IntEnum):
    """Field type enum matching C++ implementation"""
    VELOCITY = 0
    PRESSURE = 1
    GRADIENT = 2
    ADJOINT = 3
    RESIDUAL = 4


class SnapshotData:
    """Container for snapshot data and metadata"""
    def __init__(self, data: np.ndarray, nx: int, ny: int, nz: int,
                 dx: float, dy: float, dz: float, timestep: int,
                 dt: float, field_type: FieldType):
        self.data = data
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.timestep = timestep
        self.dt = dt
        self.field_type = field_type

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Grid dimensions (nx, ny, nz)"""
        return (self.nx, self.ny, self.nz)

    @property
    def extent_xz(self) -> Tuple[float, float, float, float]:
        """Extent for matplotlib imshow: (x_min, x_max, z_min, z_max)"""
        return (0, self.nx * self.dx, self.nz * self.dz, 0)

    @property
    def is_2d(self) -> bool:
        """Check if this is a 2D model (ny == 1)"""
        return self.ny == 1

    def get_2d_slice(self, y_index: int = 0) -> np.ndarray:
        """
        Extract 2D slice for visualization.
        For 2D models (ny=1), returns shape (nz, nx) for proper orientation.
        For 3D models, extracts the y-plane at y_index.
        """
        if self.is_2d:
            # Reshape and transpose for proper orientation (Z vertical, X horizontal)
            return self.data.reshape(self.nx, self.ny, self.nz)[:, 0, :].T
        else:
            # Extract Y-slice
            return self.data.reshape(self.nx, self.ny, self.nz)[:, y_index, :].T


def read_snapshot(filename: str) -> SnapshotData:
    """
    Read a binary snapshot file.

    Args:
        filename: Path to .bin snapshot file

    Returns:
        SnapshotData object containing data and metadata

    Raises:
        ValueError: If file format is invalid
        IOError: If file cannot be read
    """
    with open(filename, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != b'SWAV':
            raise ValueError(f"Invalid file format. Expected magic 'SWAV', got {magic}")

        version = struct.unpack('I', f.read(4))[0]
        if version != 1:
            raise ValueError(f"Unsupported version {version}")

        # Grid dimensions
        nx, ny, nz = struct.unpack('III', f.read(12))

        # Grid spacing
        dx, dy, dz = struct.unpack('fff', f.read(12))

        # Time info
        timestep = struct.unpack('I', f.read(4))[0]
        dt = struct.unpack('f', f.read(4))[0]

        # Field type
        field_type = FieldType(struct.unpack('I', f.read(4))[0])

        # Read data
        data_size = nx * ny * nz
        data = np.fromfile(f, dtype=np.float32, count=data_size)

        if data.size != data_size:
            raise IOError(f"Expected {data_size} floats, got {data.size}")

        return SnapshotData(data, nx, ny, nz, dx, dy, dz, timestep, dt, field_type)


def plot_velocity_model(snapshot: SnapshotData, output_file: Optional[str] = None,
                        cmap: str = 'jet', vmin: Optional[float] = None,
                        vmax: Optional[float] = None):
    """
    Plot velocity model as 2D image.

    Args:
        snapshot: SnapshotData object
        output_file: If provided, save figure to this path
        cmap: Matplotlib colormap name
        vmin, vmax: Color scale limits (auto if None)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get 2D slice
    data_2d = snapshot.get_2d_slice()

    # Plot
    im = ax.imshow(data_2d, aspect='auto', cmap=cmap,
                   extent=snapshot.extent_xz, vmin=vmin, vmax=vmax)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_title('Velocity Model (m/s)', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Velocity (m/s)', fontsize=12)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved velocity model to {output_file}")
    else:
        plt.show()


def plot_wavefield_snapshot(snapshot: SnapshotData, output_file: Optional[str] = None,
                            cmap: str = 'seismic', vmin: Optional[float] = None,
                            vmax: Optional[float] = None):
    """
    Plot a single wavefield snapshot.

    Args:
        snapshot: SnapshotData object
        output_file: If provided, save figure to this path
        cmap: Matplotlib colormap name (seismic is good for wavefields)
        vmin, vmax: Color scale limits (auto if None)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get 2D slice
    data_2d = snapshot.get_2d_slice()

    # Debug info
    print(f"Wavefield statistics:")
    print(f"  Shape: {data_2d.shape}")
    print(f"  Min: {data_2d.min():.6e}, Max: {data_2d.max():.6e}")
    print(f"  Mean: {data_2d.mean():.6e}, Std: {data_2d.std():.6e}")
    print(f"  Non-zero values: {np.count_nonzero(data_2d)} / {data_2d.size}")

    # Use symmetric color scale around zero if not specified
    if vmin is None and vmax is None:
        abs_max = np.abs(data_2d).max()
        if abs_max == 0:
            print("WARNING: All wavefield values are zero!")
            abs_max = 1.0  # Avoid division by zero in colorbar
        vmin, vmax = -abs_max, abs_max
        print(f"  Color scale: [{vmin:.6e}, {vmax:.6e}]")

    # Plot
    im = ax.imshow(data_2d, aspect='auto', cmap=cmap,
                   extent=snapshot.extent_xz, vmin=vmin, vmax=vmax)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)

    time_sec = snapshot.timestep * snapshot.dt
    ax.set_title(f'Pressure Field at t={time_sec:.4f}s (timestep {snapshot.timestep})',
                fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pressure Amplitude', fontsize=12)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved wavefield snapshot to {output_file}")
    else:
        plt.show()


def create_wavefield_animation(snapshot_dir: str, shot_id: int,
                               output_file: str, fps: int = 10,
                               cmap: str = 'seismic'):
    """
    Create animation from wavefield snapshots.

    Args:
        snapshot_dir: Base directory containing shot_XXX subdirectories
        shot_id: Shot number to animate
        output_file: Output video file (e.g., 'animation.mp4' or 'animation.gif')
        fps: Frames per second
        cmap: Matplotlib colormap
    """
    # Find all pressure snapshots for this shot
    shot_dir = Path(snapshot_dir) / f"shot_{shot_id:03d}"
    snapshot_files = sorted(shot_dir.glob("pressure_*.bin"))

    if not snapshot_files:
        raise FileNotFoundError(f"No pressure snapshots found in {shot_dir}")

    print(f"Found {len(snapshot_files)} snapshots for shot {shot_id}")

    # Read first snapshot to get dimensions and determine color scale
    first_snap = read_snapshot(str(snapshot_files[0]))
    all_data = [first_snap.get_2d_slice()]

    # Read remaining snapshots
    for snap_file in snapshot_files[1:]:
        snap = read_snapshot(str(snap_file))
        all_data.append(snap.get_2d_slice())

    # Determine global color scale
    abs_max = max(np.abs(data).max() for data in all_data)
    vmin, vmax = -abs_max, abs_max

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Initial plot
    im = ax.imshow(all_data[0], aspect='auto', cmap=cmap,
                   extent=first_snap.extent_xz, vmin=vmin, vmax=vmax,
                   animated=True)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)

    title = ax.set_title('', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pressure Amplitude', fontsize=12)

    plt.tight_layout()

    # Animation update function
    def update(frame):
        snap = read_snapshot(str(snapshot_files[frame]))
        data_2d = snap.get_2d_slice()
        im.set_array(data_2d)
        time_sec = snap.timestep * snap.dt
        title.set_text(f'Shot {shot_id}: Pressure Field at t={time_sec:.4f}s (timestep {snap.timestep})')
        return [im, title]

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(snapshot_files),
                        interval=1000/fps, blit=True, repeat=True)

    # Save animation - try different writers based on file extension and availability
    output_path = Path(output_file)
    ext = output_path.suffix.lower()

    try:
        if ext == '.gif':
            # GIF format - use Pillow
            print("Saving as GIF (using Pillow)...")
            anim.save(output_file, writer='pillow', fps=fps)
        elif ext == '.mp4':
            # MP4 format - requires ffmpeg
            try:
                print("Saving as MP4 (using ffmpeg)...")
                anim.save(output_file, writer='ffmpeg', fps=fps, bitrate=2000)
            except Exception as e:
                print(f"Warning: ffmpeg not available ({e})")
                print("Falling back to GIF format...")
                gif_file = output_path.with_suffix('.gif')
                anim.save(str(gif_file), writer='pillow', fps=fps)
                print(f"Animation saved to {gif_file} (GIF format)")
                print("\nTo save as MP4, install ffmpeg:")
                print("  macOS: brew install ffmpeg")
                print("  Ubuntu: sudo apt install ffmpeg")
                plt.close()
                return
        else:
            # Unknown format, try ffmpeg first, fallback to GIF
            print(f"Warning: Unknown format {ext}, trying ffmpeg...")
            try:
                anim.save(output_file, writer='ffmpeg', fps=fps, bitrate=2000)
            except Exception:
                print("ffmpeg not available, saving as GIF instead...")
                gif_file = output_path.with_suffix('.gif')
                anim.save(str(gif_file), writer='pillow', fps=fps)
                print(f"Animation saved to {gif_file}")
                plt.close()
                return

        print(f"Animation saved to {output_file}")
    finally:
        plt.close()


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize StarFWI snapshots')
    parser.add_argument('snapshot_dir', help='Directory containing snapshot files')
    parser.add_argument('--velocity', action='store_true',
                       help='Plot velocity model')
    parser.add_argument('--snapshot', type=int, metavar='SHOT_ID',
                       help='Plot single wavefield snapshot for given shot')
    parser.add_argument('--timestep', type=int, default=0,
                       help='Timestep to plot (with --snapshot)')
    parser.add_argument('--animate', type=int, metavar='SHOT_ID',
                       help='Create animation for given shot')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for animation')

    args = parser.parse_args()

    if args.velocity:
        # Plot velocity model
        vel_file = Path(args.snapshot_dir) / 'velocity_model.bin'
        snap = read_snapshot(str(vel_file))
        plot_velocity_model(snap, args.output)

    elif args.snapshot is not None:
        # Plot single wavefield snapshot
        snap_file = Path(args.snapshot_dir) / f"shot_{args.snapshot:03d}" / f"pressure_{args.timestep:04d}.bin"
        snap = read_snapshot(str(snap_file))
        plot_wavefield_snapshot(snap, args.output)

    elif args.animate is not None:
        # Create animation
        if not args.output:
            args.output = f"shot_{args.animate:03d}_animation.mp4"
        create_wavefield_animation(args.snapshot_dir, args.animate, args.output, args.fps)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
