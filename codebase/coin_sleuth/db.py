import os
import argparse
from itertools import product

import numpy as np
import pandas as pd


def get_run_counts(flip_sequence):
    if not flip_sequence:
        return []

    N = len(flip_sequence)
    counts = [0] * N
    current_run_length = 1
    current_char = flip_sequence[0]

    for i in range(1, N):
        if flip_sequence[i] == current_char:
            current_run_length += 1
        else:
            if current_run_length <= N:
                counts[current_run_length - 1] += 1
            current_run_length = 1
            current_char = flip_sequence[i]

    if current_run_length <= N:
        counts[current_run_length - 1] += 1

    return counts


def export_to_csv(N, title, df, export_path):
    if export_path is not None:
        folder_path = os.path.join(export_path, f'{N}')
        os.makedirs(folder_path, exist_ok=True)
        file_name = f'{title}_df_{N}.csv'
        df.to_csv(os.path.join(folder_path, file_name), index=False)


def build_observations_df(N, export_path=None):
    # Generate all binary sequences of length N
    sequences = [''.join(seq) for seq in product('01', repeat=N)]

    # Calculate run counts for each sequence
    data = []
    for seq in sequences:
        run_counts = get_run_counts(seq)
        data.append([seq] + run_counts)

    # Create DataFrame
    columns = ['key'] + [f'runs_{i+1}' for i in range(N)]
    observations_df = pd.DataFrame(data, columns=columns)

    # Export to CSV
    export_to_csv(N, 'observations', observations_df, export_path)

    return observations_df


def build_expectations_df(observations_df, export_path=None):
    # Extract sequence length N
    N = len(observations_df.iloc[0]['key'])

    # Calculate the expected counts for each run length up to 'max_length'
    expected_counts = []
    for run_length in range(1, N + 1):
        # Calculate the mean count for the current run length
        mean_count = observations_df[f'runs_{run_length}'].mean()
        expected_counts.append(mean_count)

    # Create a new DataFrame for the expectation data
    # (there will be only one row in this DataFrame)
    columns = ['length'] + [f'expected_runs_{i+1}' for i in range(N)]
    expectations_df = pd.DataFrame([[N] + expected_counts], columns=columns)

    # Export to CSV
    export_to_csv(N, 'expectations', expectations_df, export_path)

    return expectations_df


def build_statistics_df(observations_df, expectations_df, export_path=None):
    # Extract sequence length N
    N = len(observations_df.iloc[0]['key'])

    # Extract expected counts from the expectations_df
    expected_counts = expectations_df.iloc[0, 1:].values.astype(float)

    # Prepare to store 𝜒^2 values
    chi_squared_results = []

    # Iterate over each row in observations_df to calculate 𝜒^2 values
    for index, row in observations_df.iterrows():
        observed_counts = row[1:].values.astype(float)
        # Calculate the 𝜒^2 statistic
        chi_squared = np.sum(
            (observed_counts - expected_counts)**2 / expected_counts)
        chi_squared_results.append((row['key'], chi_squared))

    # Create a DataFrame from the results
    statistics_df = pd.DataFrame(chi_squared_results,
                                 columns=['key', 'chi_squared'])

    # Calculate p-values as the proportion of all 𝜒^2 values 
    # that are <= each observed 𝜒^2 value
    all_chi_squared_values = statistics_df['chi_squared'].values
    statistics_df['p_value'] = [
        np.mean(all_chi_squared_values >= x) for x in all_chi_squared_values
    ]

    # Export to CSV
    export_to_csv(N, 'statistics', statistics_df, export_path)

    return statistics_df


def build_statistics_database(N,
                              filename='statistics_database.h5',
                              csv_export=False,
                              verbose=False):
    # Define the path to the HDF5 file within the 'data' folder
    folder_path = 'data'
    file_path = os.path.join(folder_path, filename)

    # Ensure the 'data' directory exists
    os.makedirs(folder_path, exist_ok=True)

    # Initialize or open the HDF5 file
    with pd.HDFStore(
            file_path,
            'a') as store:  # 'a' for read/write if it exists, create otherwise
        # Determine which dataframes already exist
        existing_keys = set(store.keys())
        expected_keys = {f'/statistics/length_{n}' for n in range(1, N + 1)}

        # Find out which lengths need to be added or updated
        keys_to_add = expected_keys - existing_keys

        # Build and add the DataFrames for the new lengths
        for key in keys_to_add:
            length = int(key.split('_')[-1])
            if verbose:
                print(
                    f"Building and storing statistics for length {length}...")

            # Set CSV export path
            csv_export_path = folder_path if csv_export else None

            # Build the observations DataFrame
            observations_df = build_observations_df(length, csv_export_path)
            if verbose:
                print(
                    f"\tObservations DataFrame for length {length} built and saved."
                )

            # Build the expectations DataFrame
            expectations_df = build_expectations_df(observations_df,
                                                    csv_export_path)
            if verbose:
                print(
                    f"\tExpectations DataFrame for length {length} built and saved."
                )

            # Build the statistics DataFrame
            statistics_df = build_statistics_df(observations_df,
                                                expectations_df,
                                                csv_export_path)
            if verbose:
                print(
                    f"\tStatistics DataFrame for length {length} built and saved.\n"
                )

            # Store the DataFrame in the HDF5 file with a key corresponding to the string length
            store.put(key, statistics_df, format='table', data_columns=True)

    print("Statistics database build complete!")
    

def run():
    parser = argparse.ArgumentParser(description='Build a statistics database for binary strings.')
    parser.add_argument('depth', type=int, help='The depth of statistics to build.')
    parser.add_argument('--csv', action='store_true', help='Enable CSV export.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')

    args = parser.parse_args()

    build_statistics_database(args.depth, csv_export=args.csv, verbose=args.verbose)

if __name__ == '__main__':
    run()

