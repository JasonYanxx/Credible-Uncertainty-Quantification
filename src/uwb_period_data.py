import csv
from collections import defaultdict
from typing import Dict

import numpy as np


def load_uwb_period_data(csv_path: str) -> Dict[int, Dict]:
    """
    Load UWB trajectory/covariance data and group measurements by period_id.

    Supports both 2D and 3D CSV layouts.
    """
    print(f"Loading UWB data from {csv_path}...")
    periods_data = defaultdict(list)

    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        sample_row = next(reader, None)
        if sample_row is None:
            print("Warning: CSV file is empty")
            return periods_data

        is_3d = (
            'estimated_z' in sample_row and
            'true_z' in sample_row and
            'claimed_cov_02' in sample_row and
            'claimed_cov_20' in sample_row and
            'claimed_cov_22' in sample_row
        )

        file.seek(0)
        reader = csv.DictReader(file)
        for row in reader:
            period_id = int(row['period_id'])
            try:
                if is_3d:
                    measurement = {
                        'packet_idx': int(row['packet_idx']),
                        'estimated_pos': np.array([
                            float(row['estimated_x']),
                            float(row['estimated_y']),
                            float(row['estimated_z']),
                        ]),
                        'true_pos': np.array([
                            float(row['true_x']),
                            float(row['true_y']),
                            float(row['true_z']),
                        ]),
                        'claimed_cov': np.array([
                            [float(row['claimed_cov_00']), float(row['claimed_cov_01']), float(row['claimed_cov_02'])],
                            [float(row['claimed_cov_10']), float(row['claimed_cov_11']), float(row['claimed_cov_12'])],
                            [float(row['claimed_cov_20']), float(row['claimed_cov_21']), float(row['claimed_cov_22'])],
                        ]),
                    }
                else:
                    measurement = {
                        'packet_idx': int(row['packet_idx']),
                        'estimated_pos': np.array([float(row['estimated_x']), float(row['estimated_y'])]),
                        'true_pos': np.array([float(row['true_x']), float(row['true_y'])]),
                        'claimed_cov': np.array([
                            [float(row['claimed_cov_00']), float(row['claimed_cov_01'])],
                            [float(row['claimed_cov_10']), float(row['claimed_cov_11'])],
                        ]),
                    }
                periods_data[period_id].append(measurement)
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping row due to error: {e}")
                continue

    print(f"Loaded {'3D' if is_3d else '2D'} data for {len(periods_data)} periods")
    for period_id, measurements in periods_data.items():
        print(f"  Period {period_id}: {len(measurements)} measurements")
    return periods_data
