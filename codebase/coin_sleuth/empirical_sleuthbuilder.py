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
def set_comp_limit(comp_limit):
    global COMP_LIMIT
    COMP_LIMIT = comp_limit

DB_FOLDER_PATH = 'data'
def set_db__folder_path(folder_path):
    global DB_FOLDER_PATH
    DB_FOLDER_PATH = folder_path

DB_FILE_NAME = 'statistics_database.h5'
def set_db_file_name(file_name):
    global DB_FILE_NAME
    DB_FILE_NAME = file_name


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


def get_chi_squared(observed, expected, 
                    verbose=False):
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


def get_db_path():
    # Ensure the folder_path directory exists
    os.makedirs(DB_FOLDER_PATH, exist_ok=True)
    # Returned the full database path
    return os.path.join(DB_FOLDER_PATH, DB_FILE_NAME)


def build_database(lower_bound, upper_bound, 
                              store_observations=False, 
                              store_expectations=False,
                              store_statistics=False, 
                              verbose=False):
    # Initialize or open the HDF5 databse
    with pd.HDFStore(get_db_path(), 'a') as store:  # 'a' for read/write if it exists, create otherwise
        # Determine which dataframes already exist
        existing_keys = set(store.keys())

        # Database key generation
        def get_key(n, df_name):
            return f'/{df_name}/length_{n}'

        # Build and add the dataframes
        for n in range(lower_bound, upper_bound + 1):

            statistics_key   = get_key(n, 'statistics')
            expectations_key = get_key(n, 'expectations')
            observations_key = get_key(n, 'observations')

            have_statistics   = statistics_key in existing_keys
            have_expectations = expectations_key in existing_keys
            have_observations = observations_key in existing_keys

            need_statistics   = store_statistics and not have_statistics
            need_expectations = (store_expectations and not have_expectations) or need_statistics
            need_observations = (store_observations and not have_observations) or need_expectations or need_statistics 

            if verbose:
                print(f"\nGetting DataFrames for length_{n}...")

            if need_observations:
                if have_observations:
                    observations_df = store[observations_key]
                else:
                    observations_df = build_observations_df(n)
                    if store_observations:
                        store.put(observations_key, observations_df, format='table', data_columns=True)
            if verbose and store_observations:
                print(f"\tObservations for length_{n} {'found' if have_observations else 'generated'}.")

            if need_expectations:
                if have_expectations:
                    expectations_df = store[expectations_key]
                else:
                    expectations_df = build_expectations_df(observations_df) 
                    if store_expectations:
                        store.put(expectations_key, expectations_df, format='table', data_columns=True)
            if verbose and store_expectations:
                print(f"\tExpectations for length_{n} {'found' if have_expectations else 'generated'}.")


            if need_statistics:
                statistics_df = build_statistics_df(observations_df, expectations_df)
                store.put(statistics_key, statistics_df, format='table', data_columns=True)
            if verbose and store_statistics:
                print(f"\tStatistics for length_{n} {'found' if have_statistics else 'generated'}.")    


    print("\nDatabase build complete!")


def get_statistics(sequence, 
                   save_new_data=False, 
                   verbose=False):
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
                print(f"Expected counts for {id} not found. Generating now...")
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
                print(f"Statistics for {id} not found. Generating now...")
            observations_df = build_observations_df(N)
            statistics_df = build_statistics_df(observations_df, expectations_df)
            if save_new_data:
                store.put(statistics_key, statistics_df, format='table', data_columns=True)

        # Find p-value range
        lower_p_values = statistics_df['p_value'][statistics_df['chi_squared'] >= chi_squared]
        upper_p_values = statistics_df['p_value'][statistics_df['chi_squared'] <= chi_squared]
        p_value_range = (lower_p_values.max(), upper_p_values.min())

        return {"chi_squared" : chi_squared, "p_value_range" : p_value_range}


print("Builder ready...\n")
