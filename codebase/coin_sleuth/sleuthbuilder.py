import os
# import argparse
from itertools import product

import numpy as np
import pandas as pd

# Conventions:
# - A 'sequence' is a binary string that represents a sequence of coin flips. 
# - The variable 'N' reprents the length of the sequences under consideration. 


def get_run_counts(sequence):
    if not sequence:
        return []

    N = len(sequence)
    counts = [0] * N
    current_run_length = 1
    current_char = sequence[0]

    for i in range(1, N):
        if sequence[i] == current_char:
            current_run_length += 1
        else:
            if current_run_length <= N:
                counts[current_run_length - 1] += 1
            current_run_length = 1
            current_char = sequence[i]

    if current_run_length <= N:
        counts[current_run_length - 1] += 1

    return counts


def build_observations_df(N, COMP_LIMIT=20):
    # Check if N exceeds the computational limit to decide on the generation method
    if N > COMP_LIMIT:
        # Use random sampling for large N
        sequences = [''.join(np.random.choice(['0', '1'], N)) for _ in range(2**COMP_LIMIT)]
    else:
        # Generate all possible binary sequences for N <= 20
        sequences = [''.join(seq) for seq in product('01', repeat=N)]

    # Calculate run counts for each sequence
    data = []
    for seq in sequences:
        run_counts = get_run_counts(seq)
        data.append([seq] + run_counts)

    # Create DataFrame
    columns = ['key'] + [f'runs_{i+1}' for i in range(N)]
    observations_df = pd.DataFrame(data, columns=columns)

    return observations_df


def build_expectations_df(observations_df):
    # Extract sequence length N
    N = len(observations_df.iloc[0]['key'])

    # Expected run count floor value:
    # This would only be used if N > COMP_LIMIT and observations are sampled
    MIN_EXPECTATION = (0.5)**(N)

    # Calculate the expected counts for each run length up to 'max_length'
    expected_counts = []
    for run_length in range(1, N + 1):
        # Calculate the mean count for the current run length
        mean_count = max(observations_df[f'runs_{run_length}'].mean(), MIN_EXPECTATION)
        expected_counts.append(mean_count)

    # Create a new DataFrame for the expectation data
    # (there will be only one row in this DataFrame)
    columns = ['length'] + [f'exp_runs_{i+1}' for i in range(N)]
    expectations_df = pd.DataFrame([[N] + expected_counts], columns=columns)

    return expectations_df


def build_statistics_df(observations_df, expectations_df):
    # Extract expected counts from the expectations_df
    expected_counts = expectations_df.iloc[0, 1:].values.astype(float)

    # Prepare to store chi-squared values
    chi_squared_results = []

    # Iterate over each row in observations_df to calculate chi-squared values
    for index, row in observations_df.iterrows():
        observed_counts = row[1:].values.astype(float)
        # Calculate the chi-squared statistic
        chi_squared = np.sum((observed_counts - expected_counts)**2 / expected_counts)
        chi_squared_results.append((row['key'], chi_squared))

    # Create a DataFrame from the results
    statistics_df = pd.DataFrame(chi_squared_results, columns=['key', 'chi_squared'])

    # Sort the DataFrame by chi-squared values in descending order
    statistics_df.sort_values('chi_squared', ascending=False, inplace=True)
    statistics_df.reset_index(drop=True, inplace=True)

    # Calculate p-values using the method described
    p_values = np.zeros(len(statistics_df))
    j = 0  # Flag to mark the start of a new block of distinct chi-squared values
    total = len(statistics_df)  # Total number of observations

    # Iterate over the sorted DataFrame
    for i in range(total):
        if i == total - 1 or statistics_df.loc[i, 'chi_squared'] > statistics_df.loc[i + 1, 'chi_squared']:
            # Calculate p-value for the block
            p_value = (i + 1) / total
            p_values[j:i+1] = p_value
            j = i + 1  # Move the start of the next block to the next element

    statistics_df['p_value'] = p_values

    return statistics_df


def build_statistics_database(lower, upper,
                              filename='statistics_database.h5',
                              store_observations=False,
                              store_expectations=False,
                              verbose=False):
    # Define the path to the HDF5 file within the 'data' folder
    folder_path = 'data'
    file_path = os.path.join(folder_path, filename)

    # Ensure the 'data' directory exists
    os.makedirs(folder_path, exist_ok=True)

    # Initialize or open the HDF5 file
    with pd.HDFStore(file_path, 'a') as store:  # 'a' for read/write if it exists, create otherwise

        # Determine which dataframes already exist
        existing_keys = set(store.keys())

        def get_key(n, df_name):
            return f'/{df_name}/length_{n}'

        # Build and add the DataFrames
        for n in range(lower, upper + 1):
            statistics_key = get_key(n, 'statistics')

            # Check if statistics have already been computed
            if statistics_key in existing_keys:
                print(f'Found key: {statistics_key}')
                continue
            if verbose:
                print(f"Building and storing statistics for {n}-strings...")

            # Build the observations DataFrame
            observations_df = build_observations_df(n)
            if store_observations:
                observations_key = get_key(n, 'observations')
                if observations_key not in existing_keys:
                    store.put(observations_key, observations_df, format='table', data_columns=True)
            if verbose:
                print(f"\tObservations DataFrame for {n}-strings built and saved.")

            # Build the expectations DataFrame
            expectations_df = build_expectations_df(observations_df)
            if store_expectations:
                expectations_key = get_key(n, 'expectations')
                if expectations_key not in existing_keys:
                    store.put(expectations_key, expectations_df, format='table', data_columns=True)
            if verbose:
                print(f"\tExpectations DataFrame for {n}-strings built and saved.")

            # Build the statistics DataFrame
            statistics_df = build_statistics_df(observations_df, expectations_df)
            store.put(statistics_key, statistics_df, format='table', data_columns=True)
            if verbose:
                print(f"\tStatistics DataFrame for {n}-strings built and saved.\n")


    print("Statistics database build complete!")


def get_p_value_for_sequence(sequence, file_path='data/statistics_database.h5'):
    
    # Determine the length of the binary string to find the corresponding dataset
    length = len(sequence)
    key = f'/statistics/length_{length}'

    # Open the HDF5 file and read the specific dataset
    with pd.HDFStore(file_path, 'r') as store:
        if key in store:
            df = store[key]
            # Check if the binary string is in the DataFrame
            result = df[df['key'] == sequence]
            if not result.empty:
                return result['p_value'].values[0]
            else:
                return "String not found in the database."
        else:
            return "No data available for strings of this length."


# def run():
#     parser = argparse.ArgumentParser(description='Build a statistics database for binary strings.')
#     parser.add_argument('depth', type=int, help='The depth of statistics to build.')
#     parser.add_argument('--csv', action='store_true', help='Enable CSV export.')
#     parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')

#     args = parser.parse_args()

#     build_statistics_database(args.depth, csv_export=args.csv, verbose=args.verbose)

# if __name__ == '__main__':
#     run()

print("Builder ready...\n")
