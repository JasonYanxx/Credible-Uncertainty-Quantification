#!/usr/bin/env python3
"""
Plot Ground Truth Rig Position Time Series
Plots the x, y, z coordinates of the rig over time from UWB dataset
"""

import argparse
import csv
import math
import os
from typing import List, Tuple
import statistics


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATASET = os.path.join(PROJECT_ROOT, "data", "external", "starloc_data_grid_s3_uwb.csv")
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

class GroundTruthPlotter:
    """
    Class to plot ground truth rig position time series
    """
    
    def __init__(self, csv_file_path: str):
        """
        Initialize the plotter with UWB data
        
        Args:
            csv_file_path: Path to the CSV file containing UWB measurements
        """
        self.csv_file_path = csv_file_path
        self.data = []
        self.timestamps = []
        self.positions_x = []
        self.positions_y = []
        self.positions_z = []
        self.rotations_w = []
        self.rotations_x = []
        self.rotations_y = []
        self.rotations_z = []
        
    def load_data(self):
        """Load and extract ground truth position data"""
        print("Loading ground truth position data...")
        
        with open(self.csv_file_path, 'r') as file:
            reader = csv.DictReader(file)
            self.data = list(reader)
        
        # Extract ground truth positions and rotations
        for row in self.data:
            self.timestamps.append(float(row['time_s']))
            self.positions_x.append(float(row['x']))
            self.positions_y.append(float(row['y']))
            self.positions_z.append(float(row['z']))
            self.rotations_w.append(float(row['w']))
            self.rotations_x.append(float(row['rot_x']))
            self.rotations_y.append(float(row['rot_y']))
            self.rotations_z.append(float(row['rot_z']))
        
        print(f"Loaded {len(self.data)} measurements")
        print(f"Time range: {min(self.timestamps):.3f} to {max(self.timestamps):.3f} seconds")
        print(f"Duration: {max(self.timestamps) - min(self.timestamps):.3f} seconds")
        
    def identify_static_periods(self, velocity_threshold: float = 0.01, min_duration: float = 0.5):
        """
        Identify periods where the rig is static using 3D velocity
        
        Args:
            velocity_threshold: Maximum allowed 3D velocity (m/s) to consider as static
            min_duration: Minimum duration (seconds) for a period to be considered static
            
        Returns:
            List of tuples (start_time, end_time, duration) for static periods
        """
        print(f"\nIdentifying static periods using 3D velocity (threshold={velocity_threshold} m/s, min_duration={min_duration}s)...")
        
        # Calculate 3D velocities
        velocities_3d = []
        for i in range(len(self.timestamps)):
            if i == 0:
                velocities_3d.append(0.0)
            else:
                dt = self.timestamps[i] - self.timestamps[i-1]
                if dt > 0:
                    dx = self.positions_x[i] - self.positions_x[i-1]
                    dy = self.positions_y[i] - self.positions_y[i-1]
                    dz = self.positions_z[i] - self.positions_z[i-1]
                    velocity_3d = math.sqrt(dx*dx + dy*dy + dz*dz) / dt
                    velocities_3d.append(velocity_3d)
                else:
                    velocities_3d.append(0.0)
        
        static_periods = []
        current_static_start = None
        current_static_count = 0
        
        for i in range(len(self.timestamps)):
            velocity_3d = velocities_3d[i]
            
            # Check if velocity is below threshold
            if velocity_3d <= velocity_threshold:
                # Rig is static
                if current_static_start is None:
                    current_static_start = self.timestamps[i]
                    current_static_count = 1
                else:
                    current_static_count += 1
            else:
                # Rig is moving
                if current_static_start is not None:
                    # End of static period
                    static_duration = self.timestamps[i-1] - current_static_start
                    if static_duration >= min_duration:
                        static_periods.append((current_static_start, self.timestamps[i-1], static_duration))
                    current_static_start = None
                    current_static_count = 0
        
        # Handle case where dataset ends during a static period
        if current_static_start is not None:
            static_duration = self.timestamps[-1] - current_static_start
            if static_duration >= min_duration:
                static_periods.append((current_static_start, self.timestamps[-1], static_duration))
        
        # Print results
        print(f"Found {len(static_periods)} static periods:")
        total_static_time = 0
        for i, (start_time, end_time, duration) in enumerate(static_periods):
            print(f"  Period {i+1}: {start_time:.3f}s - {end_time:.3f}s (duration: {duration:.3f}s)")
            total_static_time += duration
        
        total_time = self.timestamps[-1] - self.timestamps[0]
        static_percentage = (total_static_time / total_time) * 100

        return static_periods
     
    def create_simple_plot(self):
        """Create a simple text-based plot of the trajectory"""
        print("\n" + "="*60)
        print("2D TRAJECTORY VISUALIZATION (Text-based)")
        print("="*60)
        
        # Normalize coordinates to fit in a grid
        min_x, max_x = min(self.positions_x), max(self.positions_x)
        min_y, max_y = min(self.positions_y), max(self.positions_y)
        
        # Create a simple grid representation
        grid_width = 60
        grid_height = 20
        
        # Normalize positions to grid coordinates
        def normalize_x(x):
            return int((x - min_x) / (max_x - min_x) * (grid_width - 1))
        
        def normalize_y(y):
            return int((y - min_y) / (max_y - min_y) * (grid_height - 1))
        
        # Create grid
        grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]
        
        # Plot trajectory
        for i, (x, y) in enumerate(zip(self.positions_x, self.positions_y)):
            grid_x = normalize_x(x)
            grid_y = normalize_y(y)
            
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                if i == 0:
                    grid[grid_y][grid_x] = 'S'  # Start
                elif i == len(self.positions_x) - 1:
                    grid[grid_y][grid_x] = 'E'  # End
                else:
                    grid[grid_y][grid_x] = '.'  # Path
        
        # Print grid (flip Y axis for proper display)
        print("Y")
        print("↑")
        for row in reversed(grid):
            print("|" + "".join(row))
        print("+" + "-" * grid_width)
        print(" " + "X")
        
        print(f"\nLegend: S=Start, E=End, .=Path")
        print(f"X range: {min_x:.3f} to {max_x:.3f} m")
        print(f"Y range: {min_y:.3f} to {max_y:.3f} m")
        
        
    def save_static_periods(self, static_periods, filename: str):
        """Save static periods to CSV file"""
        print(f"\nSaving static periods to {filename}...")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', newline='') as file:
            fieldnames = ['period_id', 'start_time', 'end_time', 'duration', 'start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, (start_time, end_time, duration) in enumerate(static_periods):
                # Find corresponding positions
                start_idx = min(range(len(self.timestamps)), key=lambda i: abs(self.timestamps[i] - start_time))
                end_idx = min(range(len(self.timestamps)), key=lambda i: abs(self.timestamps[i] - end_time))
                
                writer.writerow({
                    'period_id': i + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'start_x': self.positions_x[start_idx],
                    'start_y': self.positions_y[start_idx],
                    'start_z': self.positions_z[start_idx],
                    'end_x': self.positions_x[end_idx],
                    'end_y': self.positions_y[end_idx],
                    'end_z': self.positions_z[end_idx]
                })
        
        print(f"Static periods saved to {filename}")
        
    def create_matplotlib_plot(self, static_periods=None):
        """Create plots using matplotlib (if available)"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            print("\nCreating matplotlib plots...")
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Ground Truth Rig Position Time Series', fontsize=16, fontweight='bold')
            
            # Plot 1: Position vs Time
            axes[0].plot(self.timestamps, self.positions_x, 'r-', label='X', linewidth=1)
            axes[0].plot(self.timestamps, self.positions_y, 'g-', label='Y', linewidth=1)
            axes[0].plot(self.timestamps, self.positions_z, 'b-', label='Z', linewidth=1)
            
            # Highlight static periods using patches
            if static_periods:
                from matplotlib.patches import Rectangle
                from matplotlib.collections import PatchCollection
                
                # Get the y-axis limits for the static period patches
                y_min = min(min(self.positions_x), min(self.positions_y), min(self.positions_z))
                y_max = max(max(self.positions_x), max(self.positions_y), max(self.positions_z))
                y_range = y_max - y_min
                y_padding = y_range * 0.05  # 5% padding
                
                patches = []
                for i, (start_time, end_time, duration) in enumerate(static_periods):
                    width = end_time - start_time
                    height = y_range + 2 * y_padding
                    rect = Rectangle((start_time, y_min - y_padding), width, height, 
                                   alpha=0.3, facecolor='gray', edgecolor='black', linewidth=0.5)
                    patches.append(rect)
                
                # Add patches to the plot
                patch_collection = PatchCollection(patches, match_original=True)
                axes[0].add_collection(patch_collection)
                
                # Add legend entry for static periods
                axes[0].plot([], [], color='gray', alpha=0.3, linewidth=10, label='Static Periods')
            
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Position (m)')
            axes[0].set_title('Position vs Time (Static periods highlighted)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: 2D Trajectory
            axes[1].plot(self.positions_x, self.positions_y, 'b-', linewidth=1, alpha=0.7)
            axes[1].plot(self.positions_x[0], self.positions_y[0], 'g*', markersize=8, label='Start')
            axes[1].plot(self.positions_x[-1], self.positions_y[-1], 'r*', markersize=8, label='End')
            
            # Highlight static positions in 2D trajectory
            if static_periods:
                static_x = []
                static_y = []
                for start_time, end_time, duration in static_periods:
                    # Find positions during static periods
                    for i, timestamp in enumerate(self.timestamps):
                        if start_time <= timestamp <= end_time:
                            static_x.append(self.positions_x[i])
                            static_y.append(self.positions_y[i])
                
                if static_x and static_y:
                    # transparent blue
                    axes[1].scatter(static_x, static_y, c='b', s=40, alpha=0.6, label='Static Positions')
            
            axes[1].set_xlabel('X Position (m)')
            axes[1].set_ylabel('Y Position (m)')
            axes[1].set_title('2D Trajectory (X-Y) with Static Positions')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].axis('equal')
            
            
            output_path = os.path.join(DEFAULT_RESULTS_DIR, "ground_truth_timeseries.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved as '{output_path}'")
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Install with: conda install matplotlib")
            print("Using text-based visualization instead.")
            self.create_simple_plot()

    def plot_static_periods(self, output_path: str, static_periods=None):
        """Create only the first figure (Position vs Time) using matplotlib (if available)"""
        import matplotlib.pyplot as plt
        import numpy as np

        print("\nCreating matplotlib plot (Position vs Time)...")

        # Create figure with a single subplot
        target_pixel_width = 428
        target_pixel_height = 322
        base_dpi = 100
        fig_width_in = target_pixel_width / base_dpi
        fig_height_in = target_pixel_height / base_dpi
        resolution_dpi = 300

        fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
        # Plot 1: Position vs Time
        ax.plot(self.timestamps, self.positions_x, 'r-', label='X', linewidth=1)
        ax.plot(self.timestamps, self.positions_y, 'g-', label='Y', linewidth=1)
        ax.plot(self.timestamps, self.positions_z, 'b-', label='Z', linewidth=1)

        # Highlight static periods using patches
        if static_periods:
            from matplotlib.patches import Rectangle
            from matplotlib.collections import PatchCollection

            # Get the y-axis limits for the static period patches
            y_min = min(min(self.positions_x), min(self.positions_y), min(self.positions_z))
            y_max = max(max(self.positions_x), max(self.positions_y), max(self.positions_z))
            y_range = y_max - y_min
            y_padding = y_range * 0.05  # 5% padding

            patches = []
            for i, (start_time, end_time, duration) in enumerate(static_periods):
                width = end_time - start_time
                height = y_range + 2 * y_padding
                rect = Rectangle((start_time, y_min - y_padding), width, height,
                                    alpha=0.3, facecolor='gray', edgecolor='black', linewidth=0.5)
                patches.append(rect)

            # Add patches to the plot
            patch_collection = PatchCollection(patches, match_original=True)
            ax.add_collection(patch_collection)

            # Add legend entry for static periods
            ax.plot([], [], color='gray', alpha=0.3, linewidth=10, label='Static Periods')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=resolution_dpi)
        plt.show()

def main():
    """Main function to plot ground truth time series"""
    parser = argparse.ArgumentParser(description="Extract static periods from the STARloc UWB dataset.")
    parser.add_argument("--csv", default=DEFAULT_DATASET, help="Path to the STARloc UWB CSV file")
    parser.add_argument(
        "--static-periods-out",
        default=os.path.join(DEFAULT_RESULTS_DIR, "static_periods.csv"),
        help="Output CSV path for extracted static periods",
    )
    parser.add_argument(
        "--static-xyz-figure",
        default=os.path.join(DEFAULT_RESULTS_DIR, "static_period_xyz.png"),
        help="Output path for the time-series figure used in the paper",
    )
    parser.add_argument("--velocity-threshold", type=float, default=0.1, help="Static-velocity threshold in m/s")
    parser.add_argument("--min-duration", type=float, default=4.0, help="Minimum static duration in seconds")
    args = parser.parse_args()

    plotter = GroundTruthPlotter(args.csv)
    
    # Load data and create plots
    plotter.load_data()
    
    # Identify static periods
    print("\n" + "="*60)
    print("STATIC PERIOD ANALYSIS")
    print("="*60)
    
    velocity_threshold = args.velocity_threshold
    min_duration = args.min_duration
    
    print(f"\n--- Velocity Threshold: {velocity_threshold} m/s ---")
    static_periods = plotter.identify_static_periods(velocity_threshold=velocity_threshold, min_duration=min_duration)
    
    # Save results
    plotter.save_static_periods(static_periods, args.static_periods_out)
    
    # Plot Position vs Time
    plotter.plot_static_periods(args.static_xyz_figure, static_periods=static_periods)

    # Create plots with static periods highlighted
    plotter.create_matplotlib_plot(static_periods=static_periods)
    
    print("\nGround truth time series analysis complete!")

if __name__ == "__main__":
    main()
