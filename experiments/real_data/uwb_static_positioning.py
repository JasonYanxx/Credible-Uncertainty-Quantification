#!/usr/bin/env python3
"""
3D UWB Range-based Positioning for Static Periods - Simplified Version
Uses Weighted Least Squares (WLS) with std measurements as weights
Focuses only on essential functionality: positioning analysis and visualization
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DEFAULT_UWB_CSV = os.path.join(PROJECT_ROOT, "data", "external", "starloc_data_grid_s3_uwb.csv")
DEFAULT_STATIC_PERIODS = os.path.join(DEFAULT_RESULTS_DIR, "static_periods.csv")
DEFAULT_ANCHOR_MARKERS = os.path.join(PROJECT_ROOT, "data", "external", "uwb_markers_v2.csv")

class Static2DPositioning:
    """
    3D UWB Range-based Positioning System for Static Periods
    Uses Weighted Least Squares (WLS) positioning with std measurements as weights
    """
    
    def __init__(self, uwb_csv_path: str, static_periods_csv_path: str, anchor_markers_csv_path: str):
        self.uwb_csv_path = uwb_csv_path
        self.static_periods_csv_path = static_periods_csv_path
        self.anchor_markers_csv_path = anchor_markers_csv_path
        self.uwb_data = []
        self.static_periods = []
        self.anchor_positions = {}  # Full 3D coordinates [x, y, z] for anchors
        self.results = []
        self.trajectory_data = []  # Store trajectory data for visualization
        
    def load_data(self):
        """Load all data files"""
        # Load UWB measurements
        with open(self.uwb_csv_path, 'r') as file:
            # Filter out anchor 4 (User requested: "anchor 1 means anchor 4")
            # self.uwb_data = [row for row in csv.DictReader(file) if int(row['to_id']) != 4]
            self.uwb_data = list(csv.DictReader(file))
        
        # Load anchor positions (3D: x, y, z)
        with open(self.anchor_markers_csv_path, 'r') as file:
            for row in csv.DictReader(file):
                anchor_id = int(row['anchor_id'])
                self.anchor_positions[anchor_id] = [float(row['x']), float(row['y']), float(row['z'])]
        
        # Load static periods
        with open(self.static_periods_csv_path, 'r') as file:
            self.static_periods = list(csv.DictReader(file))
        
        print(f"Loaded {len(self.uwb_data)} UWB measurements")
        print(f"Loaded {len(self.anchor_positions)} anchors (3D): {sorted(self.anchor_positions.keys())}")
        print(f"Loaded {len(self.static_periods)} static periods")
    
    def trilateration_3d_weighted(self, ranges: Dict[int, float], stds: Dict[int, float]) -> Tuple[List[float], List[List[float]]]:
        """
        3D trilateration using weighted iterative least squares method.
        Uses 3D distance for range measurements and solves for x, y, z coordinates.
        Uses inverse square of std as weights.
        
        Args:
            ranges: Dictionary of anchor_id -> range measurement (3D distance)
            stds: Dictionary of anchor_id -> standard deviation
        
        Returns: ([x, y, z], 3x3 covariance_matrix) or ([nan, nan, nan], 3x3 nan matrix) if failed
        """
        nan_result = (
            [float('nan'), float('nan'), float('nan')], 
            [[float('nan'), float('nan'), float('nan')], 
             [float('nan'), float('nan'), float('nan')], 
             [float('nan'), float('nan'), float('nan')]]
        )
        
        if len(ranges) < 3:  # Need at least 3 anchors for 3D positioning (more anchors provide better accuracy)
            return nan_result

        # Prepare anchor positions, measured ranges, and weights
        anchor_ids = [aid for aid in ranges.keys() if aid in self.anchor_positions and aid in stds]
        if len(anchor_ids) < 3:
            return nan_result
        
        anchor_positions = [self.anchor_positions[aid] for aid in anchor_ids]
        measured_ranges = [ranges[aid] for aid in anchor_ids]
        weights = [1.0 / (stds[aid] ** 2) for aid in anchor_ids]  # Inverse square of std

        # Initial guess: centroid of anchors (3D)
        pos = [
            sum(p[0] for p in anchor_positions) / len(anchor_positions),
            sum(p[1] for p in anchor_positions) / len(anchor_positions),
            sum(p[2] for p in anchor_positions) / len(anchor_positions)
        ]

        # Initialize inv_HTH to avoid UnboundLocalError if loop breaks early
        inv_HTH = [[float('nan')]*3 for _ in range(3)]

        max_iter = 20
        eps = 1e-6

        for _ in range(max_iter):
            H = []
            b = []
            W = []  # Weight matrix diagonal elements
            total_error = 0.0
            
            for i, (anchor_id, anchor_pos, r_meas, weight) in enumerate(zip(anchor_ids, anchor_positions, measured_ranges, weights)):
                dx = pos[0] - anchor_pos[0]
                dy = pos[1] - anchor_pos[1]
                dz = pos[2] - anchor_pos[2]
                
                # Calculate 3D distance: r_3D = sqrt(dx² + dy² + dz²)
                dist_3d = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                if dist_3d < 1e-8:
                    # Avoid division by zero, skip this anchor
                    continue
                
                # Jacobian for 3D distance w.r.t. x, y, z: [∂r_3D/∂x, ∂r_3D/∂y, ∂r_3D/∂z] = [dx/r_3D, dy/r_3D, dz/r_3D]
                H.append([dx / dist_3d, dy / dist_3d, dz / dist_3d])
                # Residual: measured 3D range vs computed 3D distance
                b.append([r_meas - dist_3d])
                total_error += abs(r_meas - dist_3d)
                # Weight
                W.append(weight)

            if len(H) < 3:  # Need at least 3 anchors for 3D positioning
                return nan_result

            # Weighted least squares solution: delta = (H^T W H)^-1 H^T W b
            # Apply weights to H and b
            for i in range(len(H)):
                sqrt_w = math.sqrt(W[i])
                H[i] = [H[i][0] * sqrt_w, H[i][1] * sqrt_w, H[i][2] * sqrt_w]
                b[i] = [b[i][0] * sqrt_w]
            
            # Compute H^T H (now effectively H^T W H) - 3x3 matrix
            HT = [[H[j][i] for j in range(len(H))] for i in range(3)]  # 3xN
            HTH = [
                [sum(HT[i][k] * H[k][j] for k in range(len(H))) for j in range(3)]
                for i in range(3)
            ]
            # Compute H^T b (now effectively H^T W b) - 3x1 vector
            HTb = [sum(HT[i][k] * b[k][0] for k in range(len(H))) for i in range(3)]

            # Solve for delta using matrix inverse (3x3)
            det = (HTH[0][0] * (HTH[1][1]*HTH[2][2] - HTH[1][2]*HTH[2][1]) -
                   HTH[0][1] * (HTH[1][0]*HTH[2][2] - HTH[1][2]*HTH[2][0]) +
                   HTH[0][2] * (HTH[1][0]*HTH[2][1] - HTH[1][1]*HTH[2][0]))
            
            if abs(det) < 1e-12:
                break  # Singular, cannot update

            # Compute inverse of HTH using adjugate method
            inv_HTH = [
                [(HTH[1][1]*HTH[2][2] - HTH[1][2]*HTH[2][1]) / det,
                 (HTH[0][2]*HTH[2][1] - HTH[0][1]*HTH[2][2]) / det,
                 (HTH[0][1]*HTH[1][2] - HTH[0][2]*HTH[1][1]) / det],
                [(HTH[1][2]*HTH[2][0] - HTH[1][0]*HTH[2][2]) / det,
                 (HTH[0][0]*HTH[2][2] - HTH[0][2]*HTH[2][0]) / det,
                 (HTH[0][2]*HTH[1][0] - HTH[0][0]*HTH[1][2]) / det],
                [(HTH[1][0]*HTH[2][1] - HTH[1][1]*HTH[2][0]) / det,
                 (HTH[0][1]*HTH[2][0] - HTH[0][0]*HTH[2][1]) / det,
                 (HTH[0][0]*HTH[1][1] - HTH[0][1]*HTH[1][0]) / det]
            ]
            
            delta = [
                inv_HTH[0][0]*HTb[0] + inv_HTH[0][1]*HTb[1] + inv_HTH[0][2]*HTb[2],
                inv_HTH[1][0]*HTb[0] + inv_HTH[1][1]*HTb[1] + inv_HTH[1][2]*HTb[2],
                inv_HTH[2][0]*HTb[0] + inv_HTH[2][1]*HTb[1] + inv_HTH[2][2]*HTb[2]
            ]

            pos[0] += delta[0]
            pos[1] += delta[1]
            pos[2] += delta[2]

            if math.sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]) < eps or total_error < 0.001:
                # Projection Matrix for futher analysis
                # Pm = np.array(H) @ np.array(inv_HTH) @ np.array(H).T
                break

        return pos, inv_HTH
    
    def get_packets_in_period(self, start_time: float, end_time: float) -> List[Dict]:
        """Get measurement packets within a time period, including range and std values"""
        # Get measurements in time period
        measurements = []
        for row in self.uwb_data:
            timestamp = float(row['time_s'])
            if start_time <= timestamp <= end_time:
                anchor_id = int(row['to_id'])
                range_value = float(row['range_calib'])
                std_value = float(row['std'])
                if not math.isnan(range_value) and range_value > 0 and not math.isnan(std_value) and std_value > 0:
                    measurements.append({
                        'timestamp': timestamp,
                        'anchor_id': anchor_id,
                        'range': range_value,
                        'std': std_value
                    })
        
        # Sort by timestamp
        measurements.sort(key=lambda x: x['timestamp'])
        
        # Group into packets
        packets = []
        current_packet_ranges = {}
        current_packet_stds = {}
        
        for m in measurements:
            anchor_id = m['anchor_id']
            range_value = m['range']
            std_value = m['std']
            
            # Start new packet if anchor repeats
            if anchor_id in current_packet_ranges:
                if len(current_packet_ranges) >= 3:
                    packets.append({
                        'ranges': current_packet_ranges,
                        'stds': current_packet_stds
                    })
                current_packet_ranges = {anchor_id: range_value}
                current_packet_stds = {anchor_id: std_value}
            else:
                current_packet_ranges[anchor_id] = range_value
                current_packet_stds[anchor_id] = std_value
        
        # Add final packet if valid
        if len(current_packet_ranges) >= 3:
            packets.append({
                'ranges': current_packet_ranges,
                'stds': current_packet_stds
            })
        
        return packets
    
    def get_ground_truth_3d(self, start_time: float, end_time: float) -> List[float]:
        """Get average 3D ground truth position during period"""
        for row in self.uwb_data:
            timestamp = float(row['time_s'])
            if start_time <= timestamp <= end_time:
                return [float(row['tag_pos_x']), float(row['tag_pos_y']), float(row['tag_pos_z'])]
        
        return [float('nan'), float('nan'), float('nan')]
    
    def analyze_period(self, period_data: Dict) -> Dict:
        """Analyze positioning accuracy for one static period"""
        period_id = int(period_data['period_id'])
        start_time = float(period_data['start_time'])
        end_time = float(period_data['end_time'])
        duration = float(period_data['duration'])
        
        print(f"Period {period_id}: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)")
        
        # Get measurement packets and ground truth
        packets = self.get_packets_in_period(start_time, end_time)
        ground_truth = self.get_ground_truth_3d(start_time, end_time)
        
        if not packets or any(math.isnan(x) for x in ground_truth):
            return {
                'period_id': period_id, 'duration': duration,
                'packets': 0, 'estimates': 0, 'error_3d': float('nan')
            }
        
        # Perform 3D weighted trilateration for each packet
        errors = []
        valid_estimates = 0
        
        for i, packet in enumerate(packets):
            # Use weighted least squares trilateration for 3D positioning
            estimated_pos, claimed_covariance = self.trilateration_3d_weighted(
                packet['ranges'], packet['stds']
            )
            if not any(math.isnan(x) for x in estimated_pos):
                error = math.sqrt((estimated_pos[0] - ground_truth[0])**2 + 
                                (estimated_pos[1] - ground_truth[1])**2 +
                                (estimated_pos[2] - ground_truth[2])**2)
                errors.append(error)
                valid_estimates += 1
                
                # Store trajectory data for visualization
                self.trajectory_data.append({
                    'period_id': period_id,
                    'packet_idx': i,
                    'estimated_x': estimated_pos[0],
                    'estimated_y': estimated_pos[1],
                    'estimated_z': estimated_pos[2],
                    'true_x': ground_truth[0],
                    'true_y': ground_truth[1],
                    'true_z': ground_truth[2],
                    'error_3d': error,
                    'claimed_covariance': claimed_covariance
                })
        
        # Calculate statistics
        if errors:
            avg_error = sum(errors) / len(errors)
            rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
        else:
            avg_error = rmse = float('nan')
        
        print(f"  Packets: {len(packets)}, Estimates: {valid_estimates}, Error: {avg_error:.3f}m")
        
        return {
            'period_id': period_id,
            'duration': duration,
            'packets': len(packets),
            'estimates': valid_estimates,
            'error_3d': avg_error,
            'rmse_3d': rmse,
            'ground_truth_x': ground_truth[0],
            'ground_truth_y': ground_truth[1],
            'ground_truth_z': ground_truth[2]
        }
    
    def run_analysis(self):
        """Run complete analysis for all static periods"""
        print("\n3D UWB WEIGHTED LEAST SQUARES POSITIONING ANALYSIS")
        print("=" * 60)
        
        self.results = []
        for period_data in self.static_periods:
            result = self.analyze_period(period_data)
            self.results.append(result)
    
    def save_results(self, filename: str = 'results/static_3d_results.csv'):
        """Save results to CSV"""
        with open(filename, 'w', newline='') as file:
            fieldnames = ['period_id', 'duration', 'packets', 'estimates', 
                         'error_3d', 'rmse_3d', 'ground_truth_x', 'ground_truth_y', 'ground_truth_z']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result)
        print(f"Results saved to {filename}")
    
    def plot_positioning_results(self):
        """Plot positioning results and full trajectory of original data in one figure"""
        if not self.trajectory_data:
            print("No trajectory data available for plotting")
            return

        # Create figure with a single subplot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Extract full trajectory from original data
        timestamps = []
        x_coords = []
        y_coords = []
        
        for row in self.uwb_data:
            try:
                timestamp = float(row['time_s'])
                x_coord = float(row['tag_pos_x'])
                y_coord = float(row['tag_pos_y'])
                
                timestamps.append(timestamp)
                x_coords.append(x_coord)
                y_coords.append(y_coord)
            except (ValueError, KeyError):
                continue
        
        # Plot full trajectory as a line
        ax.plot(x_coords, y_coords, 'gray', linewidth=1, alpha=0.5, label='Full Trajectory')

        # Get unique periods and colors
        periods = sorted(list(set(d['period_id'] for d in self.trajectory_data)))
        colors = plt.cm.tab10(np.linspace(0, 1, len(periods)))

        # Plot positioning results for static periods
        for i, period_id in enumerate(periods):
            period_data = [d for d in self.trajectory_data if d['period_id'] == period_id]
            period_data.sort(key=lambda x: x['packet_idx'])

            # True position (should be same for all packets in period)
            true_x_pos = period_data[0]['true_x']
            true_y_pos = period_data[0]['true_y']

            # Estimated positions
            est_x_pos = [d['estimated_x'] for d in period_data]
            est_y_pos = [d['estimated_y'] for d in period_data]

            # Plot estimated positions as empty circles
            ax.scatter(est_x_pos, est_y_pos, facecolors='none', edgecolors=colors[i], marker='o', s=30, alpha=0.8,
                       label=f'Est P{period_id}')
            
            # Plot true position
            ax.scatter(true_x_pos, true_y_pos, color=colors[i], marker='*', s=250,
                       label=f'True P{period_id}', edgecolors='black', linewidth=1)

        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        # ax.set_title('Full Trajectory with Static Period Positioning Results')
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        # Alternatives to ax.set_aspect('equal'):
        # Option 1: Use 'auto' aspect (default, axes fill the figure)
        ax.set_aspect('auto')

        plt.tight_layout()
        plt.savefig('results/uwb_positioning_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Combined visualization saved as 'results/uwb_positioning_analysis.png'.")
    
    def plot_positioning_results_3d(self):
        """Plot 3D positioning results and full trajectory of original data in one figure"""
        if not self.trajectory_data:
            print("No trajectory data available for plotting")
            return

        # Create figure with 3D subplot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Extract full trajectory from original data (3D)
        timestamps = []
        x_coords = []
        y_coords = []
        z_coords = []
        
        for row in self.uwb_data:
            try:
                timestamp = float(row['time_s'])
                x_coord = float(row['tag_pos_x'])
                y_coord = float(row['tag_pos_y'])
                z_coord = float(row['tag_pos_z'])
                
                timestamps.append(timestamp)
                x_coords.append(x_coord)
                y_coords.append(y_coord)
                z_coords.append(z_coord)
            except (ValueError, KeyError):
                continue
        
        # Plot full trajectory as a line (3D)
        ax.plot(x_coords, y_coords, z_coords, 'black', linewidth=1, alpha=0.5, label='Full Trajectory')

        # Get unique periods and colors
        periods = sorted(list(set(d['period_id'] for d in self.trajectory_data)))
        colors = plt.cm.tab10(np.linspace(0, 1, len(periods)))

        # Plot positioning results for static periods
        for i, period_id in enumerate(periods):
            period_data = [d for d in self.trajectory_data if d['period_id'] == period_id]
            period_data.sort(key=lambda x: x['packet_idx'])

            # True position (should be same for all packets in period)
            true_x_pos = period_data[0]['true_x']
            true_y_pos = period_data[0]['true_y']
            true_z_pos = period_data[0]['true_z']

            # Estimated positions
            est_x_pos = [d['estimated_x'] for d in period_data]
            est_y_pos = [d['estimated_y'] for d in period_data]
            est_z_pos = [d['estimated_z'] for d in period_data]

            # Plot estimated positions as empty circles (3D)
            ax.scatter(est_x_pos, est_y_pos, est_z_pos, facecolors='none', edgecolors=colors[i], 
                       marker='o', s=30, alpha=0.8, label=f'Est P{period_id}')
            
            # Plot true position (3D)
            ax.scatter([true_x_pos], [true_y_pos], [true_z_pos], color=colors[i], marker='*', s=250,
                       label=f'True P{period_id}', edgecolors='black', linewidth=1)

        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.grid(True, alpha=0.1)

        plt.tight_layout()
        plt.savefig('results/uwb_positioning_analysis_3d.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("3D visualization saved as 'results/uwb_positioning_analysis_3d.png'.")
    
    def plot_anchors_and_period1_3d(self):
        """Plot anchor positions and Period 1 ground truth position in 3D"""
        if not self.anchor_positions:
            print("No anchor positions available for plotting")
            return
        
        # Get Period 1 ground truth from results
        period1_gt = None
        for result in self.results:
            if result['period_id'] == 1:
                period1_gt = [
                    result['ground_truth_x'],
                    result['ground_truth_y'],
                    result['ground_truth_z']
                ]
                break
        
        if period1_gt is None:
            print("Period 1 ground truth not found in results")
            return
        
        # Create 3D figure
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract anchor positions
        anchor_ids = sorted(self.anchor_positions.keys())
        anchor_x = [self.anchor_positions[aid][0] for aid in anchor_ids]
        anchor_y = [self.anchor_positions[aid][1] for aid in anchor_ids]
        anchor_z = [self.anchor_positions[aid][2] for aid in anchor_ids]
        
        # Plot anchors
        ax.scatter(anchor_x, anchor_y, anchor_z, c='blue', marker='s', s=100, 
                   alpha=0.8, label='Anchors', edgecolors='black', linewidth=1)
        
        # Label anchors
        for aid, x, y, z in zip(anchor_ids, anchor_x, anchor_y, anchor_z):
            ax.text(x, y, z, f' A{aid}', fontsize=8)
        
        # Plot Period 1 ground truth
        ax.scatter([period1_gt[0]], [period1_gt[1]], [period1_gt[2]], 
                   c='red', marker='*', s=500, alpha=1.0, 
                   label='Period 1 Ground Truth', edgecolors='black', linewidth=2)
        
        # Set labels and title
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.grid(True, alpha=0.1)
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig('results/anchors_and_period1_3d.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("3D visualization of anchors and Period 1 ground truth saved as 'results/anchors_and_period1_3d.png'.")
    
    def plot_std_boxplots_by_period_with_bias_calib(self):
        """Plot boxplots of 'std' values categorized by 'to_id' for each static period,
        with red stars showing standard deviation of 'bias_calib' for each anchor"""
        print("\nGenerating combined boxplots: std distributions + bias_calib std markers...")
        
        # First, calculate bias_calib statistics for all periods
        bias_calib_stats = {}
        for period_data in self.static_periods:
            period_id = int(period_data['period_id'])
            start_time = float(period_data['start_time'])
            end_time = float(period_data['end_time'])
            
            # Get bias_calib measurements for this period grouped by anchor ID
            bias_calib_by_anchor = defaultdict(list)
            for row in self.uwb_data:
                timestamp = float(row['time_s'])
                if start_time <= timestamp <= end_time:
                    anchor_id = int(row['to_id'])
                    bias_calib_value = float(row['bias_calib'])
                    if not math.isnan(bias_calib_value):
                        bias_calib_by_anchor[anchor_id].append(bias_calib_value)
            
            # Calculate std for each anchor
            period_bias_stats = {}
            for anchor_id, values in bias_calib_by_anchor.items():
                if len(values) > 1:
                    period_bias_stats[anchor_id] = np.std(values, ddof=1)
                else:
                    period_bias_stats[anchor_id] = 0.0
            
            bias_calib_stats[period_id] = period_bias_stats
        
        # Get number of static periods
        num_periods = len(self.static_periods)
        if num_periods == 0:
            print("No static periods found!")
            return
        
        # Calculate subplot layout
        cols = min(3, num_periods)  # Max 3 columns
        rows = (num_periods + cols - 1) // cols  # Ceiling division
        
        # Create figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]  # Make it iterable
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Process each static period
        for i, period_data in enumerate(self.static_periods):
            period_id = int(period_data['period_id'])
            start_time = float(period_data['start_time'])
            end_time = float(period_data['end_time'])
            
            # Get std measurements for this period grouped by anchor ID
            std_by_anchor = defaultdict(list)
            
            for row in self.uwb_data:
                timestamp = float(row['time_s'])
                if start_time <= timestamp <= end_time:
                    anchor_id = int(row['to_id'])
                    std_value = float(row['std'])
                    if not math.isnan(std_value) and std_value > 0:
                        std_by_anchor[anchor_id].append(std_value)
            
            # Prepare data for boxplot
            anchor_ids = sorted(std_by_anchor.keys())
            std_data = [std_by_anchor[aid] for aid in anchor_ids]
            anchor_labels = [f'Anchor {aid}' for aid in anchor_ids]
            
            # Create boxplot
            ax = axes[i] if len(axes) > 1 else axes
            
            if std_data and any(len(data) > 0 for data in std_data):
                # Create boxplots for std values
                bp = ax.boxplot(std_data, tick_labels=anchor_labels, patch_artist=True)
                # set ylim
                ax.set_ylim(0,0.5)
                
                # Customize boxplot colors
                colors = plt.cm.Set3(np.linspace(0, 1, len(anchor_ids)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add red stars for bias_calib standard deviations
                bias_stats = bias_calib_stats.get(period_id, {})
                for j, anchor_id in enumerate(anchor_ids):
                    if anchor_id in bias_stats:
                        bias_std = bias_stats[anchor_id]
                        # Position: x is the box position (1-based), y is the bias_calib std value
                        ax.scatter(j + 1, bias_std, marker='*', color='red', s=100, 
                                 edgecolors='darkred', linewidth=1, zorder=10,
                                 label='Empirical std' if j == 0 else "")
                
                ax.set_title(f'Period {period_id}({start_time:.1f}s - {end_time:.1f}s)')
                ax.set_ylabel('Standard Deviation')
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels if needed
                if len(anchor_ids) > 4:
                    ax.tick_params(axis='x', rotation=45)
                
                # Add legend only to the first subplot
                if i == 0:
                    # Create custom legend
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='lightblue', alpha=0.7, label='Reported std'),
                        plt.Line2D([0], [0], marker='*', color='red', linestyle='None',
                                 markersize=10, markeredgecolor='darkred', label='Empirical std')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
                
                # Add statistics text
                # total_measurements = sum(len(data) for data in std_data)
                # bias_count = len([aid for aid in anchor_ids if aid in bias_stats])
                # ax.text(0.02, 0.98, f'Std meas: {total_measurements}\nBias anchors: {bias_count}', 
                #        transform=ax.transAxes, verticalalignment='top',
                #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                #        fontsize=8)
            
            else:
                ax.text(0.5, 0.5, f'Period {period_id}\nNo valid data', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12)
                ax.set_title(f'Period {period_id} - No Data')
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/uwb_std_boxplots_with_bias_calib.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Combined boxplots saved as 'results/uwb_std_boxplots_with_bias_calib.png'")
        
    def plot_bias_calib_boxplots_by_period(self):
        """Plot boxplots of 'bias_calib' values categorized by 'to_id' for each static period"""
        print("\nGenerating boxplots of bias_calib values by anchor ID for each static period...")
        
        # Get number of static periods
        num_periods = len(self.static_periods)
        if num_periods == 0:
            print("No static periods found!")
            return
        
        # Calculate subplot layout
        cols = min(3, num_periods)  # Max 3 columns
        rows = (num_periods + cols - 1) // cols  # Ceiling division
        
        # Create figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]  # Make it iterable
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Process each static period
        for i, period_data in enumerate(self.static_periods):
            period_id = int(period_data['period_id'])
            start_time = float(period_data['start_time'])
            end_time = float(period_data['end_time'])
            
            # Get bias_calib measurements for this period grouped by anchor ID
            bias_calib_by_anchor = defaultdict(list)
            
            for row in self.uwb_data:
                timestamp = float(row['time_s'])
                if start_time <= timestamp <= end_time:
                    anchor_id = int(row['to_id'])
                    bias_calib_value = float(row['bias_calib'])
                    if not math.isnan(bias_calib_value):
                        bias_calib_by_anchor[anchor_id].append(bias_calib_value)
            
            # Prepare data for boxplot
            anchor_ids = sorted(bias_calib_by_anchor.keys())
            bias_calib_data = [bias_calib_by_anchor[aid] for aid in anchor_ids]
            anchor_labels = [f'Anchor {aid}' for aid in anchor_ids]
            
            # Create boxplot
            ax = axes[i] if len(axes) > 1 else axes
            
            if bias_calib_data and any(len(data) > 0 for data in bias_calib_data):
                bp = ax.boxplot(bias_calib_data, tick_labels=anchor_labels, patch_artist=True)
                # set ylim
                ax.set_ylim(-0.25,2.0)
                
                # Customize boxplot colors
                colors = plt.cm.viridis(np.linspace(0, 1, len(anchor_ids)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_title(f'Period {period_id} ({start_time:.1f}s - {end_time:.1f}s)')
                ax.set_ylabel('Measurement Error (m)')
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels if needed
                if len(anchor_ids) > 4:
                    ax.tick_params(axis='x', rotation=45)
                
                # Add statistics text
                total_measurements = sum(len(data) for data in bias_calib_data)
                
                # Calculate overall statistics for this period
                all_bias_values = []
                for values in bias_calib_by_anchor.values():
                    all_bias_values.extend(values)
                
                # if all_bias_values:
                    # mean_bias = np.mean(all_bias_values)
                    # std_bias = np.std(all_bias_values, ddof=1) if len(all_bias_values) > 1 else 0.0
                    
                    # ax.text(0.02, 0.98, 
                    #        f'Total: {total_measurements} meas.\n'
                    #        f'Mean: {mean_bias:.4f}\n'
                    #        f'Std: {std_bias:.4f}', 
                    #        transform=ax.transAxes, verticalalignment='top',
                    #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    #        fontsize=8)
                
                # Add horizontal line at zero for reference
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
                
            else:
                ax.text(0.5, 0.5, f'Period {period_id}\nNo valid data', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12)
                ax.set_title(f'Period {period_id} - No Data')
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/uwb_bias_calib_boxplots_by_period.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Bias calibration boxplots saved as 'results/uwb_bias_calib_boxplots_by_period.png'")
        
def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run UWB static positioning and export covariance traces.")
    parser.add_argument("--uwb-csv", default=DEFAULT_UWB_CSV, help="Path to the STARloc UWB CSV file")
    parser.add_argument(
        "--static-periods-csv",
        default=DEFAULT_STATIC_PERIODS,
        help="Path to the static-period CSV produced by static_period_extraction.py",
    )
    parser.add_argument(
        "--anchor-markers-csv",
        default=DEFAULT_ANCHOR_MARKERS,
        help="Path to the anchor metadata CSV file",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory where result CSVs and figures will be written",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Run analysis
    system = Static2DPositioning(args.uwb_csv, args.static_periods_csv, args.anchor_markers_csv)
    system.load_data()
    system.run_analysis()
    system.save_results()

    # Save trajectory data with covariance to CSV
    trajectory_output = os.path.join(args.output_dir, "uwb_trajectory_with_covariance.csv")
    with open(trajectory_output, 'w', newline='') as csvfile:
        fieldnames = [
            'period_id', 'packet_idx',
            'estimated_x', 'estimated_y', 'estimated_z',
            'true_x', 'true_y', 'true_z',
            'claimed_cov_00', 'claimed_cov_01', 'claimed_cov_02',
            'claimed_cov_10', 'claimed_cov_11', 'claimed_cov_12',
            'claimed_cov_20', 'claimed_cov_21', 'claimed_cov_22'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for d in system.trajectory_data:
            cov = d.get('claimed_covariance', [[float('nan'), float('nan'), float('nan')], 
                                                 [float('nan'), float('nan'), float('nan')], 
                                                 [float('nan'), float('nan'), float('nan')]])
            row = {
                'period_id': d.get('period_id', ''),
                'packet_idx': d.get('packet_idx', ''),
                'estimated_x': d.get('estimated_x', float('nan')),
                'estimated_y': d.get('estimated_y', float('nan')),
                'estimated_z': d.get('estimated_z', float('nan')),
                'true_x': d.get('true_x', float('nan')),
                'true_y': d.get('true_y', float('nan')),
                'true_z': d.get('true_z', float('nan')),
                'claimed_cov_00': cov[0][0] if len(cov) > 0 and len(cov[0]) > 0 else float('nan'),
                'claimed_cov_01': cov[0][1] if len(cov) > 0 and len(cov[0]) > 1 else float('nan'),
                'claimed_cov_02': cov[0][2] if len(cov) > 0 and len(cov[0]) > 2 else float('nan'),
                'claimed_cov_10': cov[1][0] if len(cov) > 1 and len(cov[1]) > 0 else float('nan'),
                'claimed_cov_11': cov[1][1] if len(cov) > 1 and len(cov[1]) > 1 else float('nan'),
                'claimed_cov_12': cov[1][2] if len(cov) > 1 and len(cov[1]) > 2 else float('nan'),
                'claimed_cov_20': cov[2][0] if len(cov) > 2 and len(cov[2]) > 0 else float('nan'),
                'claimed_cov_21': cov[2][1] if len(cov) > 2 and len(cov[2]) > 1 else float('nan'),
                'claimed_cov_22': cov[2][2] if len(cov) > 2 and len(cov[2]) > 2 else float('nan'),
            }
            writer.writerow(row)
    print(f"Trajectory with covariance saved to '{trajectory_output}'")

    # Generate visualizations
    system.plot_positioning_results()
    system.plot_positioning_results_3d()
    system.plot_anchors_and_period1_3d()
    system.plot_std_boxplots_by_period_with_bias_calib()
    system.plot_bias_calib_boxplots_by_period()

if __name__ == "__main__":
    main()
