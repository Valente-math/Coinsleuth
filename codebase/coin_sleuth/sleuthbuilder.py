import os
from itertools import product

import numpy as np
import pandas as pd

######################################## CONVENTIONS ########################################

# - A 'sequence' is a binary string that represents a sequence of coin flips.
# - The variable 'N' reprents the length of the sequences under consideration.
# - Sequences of length N are denoted 'length_N' in the database.
# - The number of runs of length L in a sequence is denoted 'runs_L' in the database.

######################################## GLOBALS ########################################

COMP_LIMIT = 10
def set_comp_limit(depth):
    global COMP_LIMIT
    COMP_LIMIT = depth

DB_FOLDER_PATH = 'data'
DB_FILE_NAME = 'statistics_database.h5'
def set_db_path(folder_path, file_name):
    global DB_FOLDER_PATH, DB_FILE_NAME
    DB_FOLDER_PATH = folder_path
    DB_FILE_NAME = file_name

def get_db_path():
    # Ensure the folder_path directory exists
    os.makedirs(DB_FOLDER_PATH, exist_ok=True)

    return os.path.join(DB_FOLDER_PATH, DB_FILE_NAME)


######################################## BUILDER METHODS ########################################


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


def build_observations_df(N):
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

    # Expected run count floor value for sampled observations:
    MIN_EXPECTATION = (0.5)**N if N > COMP_LIMIT else 0

    # Lists to store mean and standard deviation values
    means = []
    std_devs = []
    run_lengths = range(1, N + 1)

    for length in run_lengths:
        # Extract run counts for the current run length
        run_counts = observations_df[f'runs_{length}']

        # Calculate and enforce minimum expectation on the mean count
        mean = max(run_counts.mean(), MIN_EXPECTATION)
        std = run_counts.std()

        # Append results
        means.append(mean)
        std_devs.append(std)

    # Create a DataFrame
    data = {
        'run_length': run_lengths,
        'mean': means,
        'std': std_devs
    }
    expectations_df = pd.DataFrame(data)

    return expectations_df


def get_chi_squared(observed, expected, verbose=False):
    components = (observed - expected)**2 / expected
    chi_squared = np.sum(components)
    if verbose:
        print(f'Observed:\n{observed}\n')
        print(f'Expected:\n{expected}\n')
        print(f'Components:\n{components}\n-> {chi_squared}')
    return chi_squared


def build_statistics_df(observations_df, expectations_df):
    # Extract expected counts from the expectations_df
    expected_counts = expectations_df['mean'].values.astype(float)

    # Prepare to store chi-squared values
    chi_squared_results = []

    # Iterate over each row in observations_df to calculate chi-squared values
    for index, row in observations_df.iterrows():
        observed_counts = row[1:].values.astype(float)
        # Calculate the chi-squared statistic
        chi_squared = get_chi_squared(observed_counts, expected_counts)
        chi_squared_results.append((row['key'], chi_squared))

    # Create a DataFrame from the results
    statistics_df = pd.DataFrame(chi_squared_results, columns=['key', 'chi_squared'])

    # Sort the DataFrame by chi-squared values in descending order
    statistics_df.sort_values('chi_squared', ascending=False, inplace=True)
    statistics_df.reset_index(drop=True, inplace=True)

    # Total number of observations
    total = len(statistics_df) 

    # Initialize p-values to 0
    p_values = np.zeros(total)

    # Helper function for locating chi-squared values
    def chi_squared_at(index):
        return statistics_df.loc[index, 'chi_squared']

    # Iterate over the sorted DataFrame
    block_start_index  = 0  # Flag to mark the start of a new block of distinct chi-squared values
    for current_index in range(total):
        next_index = current_index + 1
        if next_index == total or chi_squared_at(current_index) > chi_squared_at(next_index):
            # Calculate p-value for the block
            p_value = (next_index) / total
            p_values[block_start_index : next_index] = p_value
            block_start_index  = next_index  # Move the start of the next block to the next element

    statistics_df['p_value'] = p_values

    return statistics_df


def build_statistics_database(lower, upper, store_observations=False, store_expectations=True, verbose=False):
    # Initialize or open the HDF5 databse
    with pd.HDFStore(get_db_path(), 'a') as store:  # 'a' for read/write if it exists, create otherwise
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


######################################## USER-FACING METHODS ########################################


def get_statistics_for_sequence(sequence, save_new_data=False, verbose=False):
    N = len(sequence)
    id = f'length_{N}'
    statistics_key = f'/statistics/{id}'
    expectations_key = f'/expectations/{id}'

    # Initialize or open the HDF5 databse
    with pd.HDFStore(get_db_path(), 'a') as store:  # 'a' to allow data addition if necessary
        # Check and load expectations data
        if expectations_key in store:
            if verbose:
                print(f"Found expected counts for {id}.")
            expectations_df = store[expectations_key]
        else:
            if verbose:
                print(f"Expectations data for {id} not found. Generating now...")
            observations_df = build_observations_df(N)
            expectations_df = build_expectations_df(observations_df)
            if save_new_data:
                store.put(expectations_key, expectations_df, format='table', data_columns=True)

        # Compute chi-squared statistic
        observed = get_run_counts(sequence)
        expected = expectations_df['mean'].values.astype(float)
        chi_squared = get_chi_squared(observed, expected)

        # Check and load statistics data
        if statistics_key in store:
            if verbose:
                print(f"Found statistics for {id}.")
            statistics_df = store[statistics_key]
        else:
            if verbose:
                print(f"Statistics data for {id} not found. Generating now...")
            observations_df = build_observations_df(N)
            statistics_df = build_statistics_df(observations_df, expectations_df)
            if save_new_data:
                store.put(statistics_key, statistics_df, format='table', data_columns=True)

        # Find p-value range
        lower_p_value = statistics_df['p_value'][statistics_df['chi_squared'] >= chi_squared]
        upper_p_value = statistics_df['p_value'][statistics_df['chi_squared'] <= chi_squared]
        p_value_range = (lower_p_value.max(), upper_p_value.min())

        return (chi_squared, p_value_range)
        # p_value = statistics_df['p_value'][statistics_df['chi_squared'] == chi_squared]
        # if verbose:
        #     print(f"Search results for p-value:\n{p_value}")

        # if p_value.empty:
        #     lower_p_value = statistics_df['p_value'][statistics_df['chi_squared'] > chi_squared]  
        #     if verbose:
        #         print(f"Search results for lower p-value:\n{lower_p_value}")

        #     upper_p_value = statistics_df['p_value'][statistics_df['chi_squared'] < chi_squared]  
        #     if verbose:
        #         print(f"Search results for upper p-value:\n{upper_p_value}")

        #     if lower_p_value.empty:
        #         if upper_p_value.empty:
        #             return f"Chi-squared: {chi_squared}, p-value not determined."
        #         else:
        #             return f"Chi-squared: {chi_squared}, p < {upper_p_value.min()}."
        #     else:
        #         if upper_p_value.empty:
        #             return f"Chi-squared: {chi_squared}, p > {lower_p_value.max()}."
        #         else:
        #             return f"Chi-squared: {chi_squared}, {lower_p_value.max()} < p < {upper_p_value.min()}"
        # else:
        #     return f"Chi-squared: {chi_squared}, p = {p_value.iloc[0]}."


print("Builder ready...\n")
