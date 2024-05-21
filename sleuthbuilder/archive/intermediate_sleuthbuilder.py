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


def get_observations(sequence):
    N = len(sequence)
    counts = [0] * N
    current_run_length = 1
    current_char = sequence[0]
    partition = []

    for i in range(1, N):
        if sequence[i] == current_char:
            # Run continues
            current_run_length += 1
        else:
            # Run ended by opposing flip
            counts[current_run_length - 1] += 1
            partition.append(current_run_length)

            # Reset run
            current_run_length = 1
            current_char = sequence[i]

    # Run ended by sequence termination
    counts[current_run_length - 1] += 1
    partition.append(current_run_length)

    return np.array(counts), np.array(partition)


def get_partition(sequence):
    N = len(sequence)
    counts = [0] * N
    current_run_length = 1
    current_char = sequence[0]
    partition = []

    for i in range(1, N):
        if sequence[i] == current_char:
            # Run continues
            current_run_length += 1
        else:
            # Run ended by opposing flip
            partition.append(current_run_length)

            # Reset run
            current_run_length = 1
            current_char = sequence[i]

    # Run ended by sequence termination
    partition.append(current_run_length)

    return np.array(partition)


def get_expectations(N):
    # Define coefficient function
    def c(n, N):
        if n >  N:
            return 0
        if n == N:
            return 1
        if n <  N:
            return 0.25*(3 + N - n)

    # Calculate expected counts
    expected_counts = [c(n, N)*(2**(1-n)) for n in range(1, N + 1)]

    return np.array(expected_counts)


def get_chi_squared(observed, expected):
    components = (observed - expected)**2 / expected
    chi_squared = np.sum(components)
    return chi_squared


def remove_common_head_and_tail(tuple_set):
    # Convert set to list to allow indexing
    tuple_set = list(tuple_set)

    # Find the common head length
    min_length = min(len(t) for t in tuple_set)
    common_head_length = 0
    for i in range(min_length):
        if all(t[i] == tuple_set[0][i] for t in tuple_set):
            common_head_length += 1
        else:
            break

    # Remove the common head
    tuple_set = [t[common_head_length:] for t in tuple_set]

    # Find the common tail length
    min_length = min(len(t) for t in tuple_set)
    common_tail_length = 0
    for i in range(1, min_length + 1):
        if all(t[-i] == tuple_set[0][-i] for t in tuple_set):
            common_tail_length += 1
        else:
            break

    # Remove the common tail
    if common_tail_length > 0:
        tuple_set = [t[:-common_tail_length] for t in tuple_set]

    return tuple_set


def analyze_block(statistics_df, block_start_index, block_end_index):
    # Check block for partition consistency
    block_id = statistics_df.loc[block_start_index, 'chi_squared']

    inconsistent_partitions = []
    current_partition = statistics_df.loc[block_start_index, 'partition']
    
    unique_partitions = {tuple(np.sort(current_partition))}
    for i in range(block_start_index + 1, block_end_index):

        next_partition = statistics_df.loc[i, 'partition']
        unique_partitions.add(tuple(np.sort(next_partition)))

        consistent = np.array_equal(np.sort(current_partition),
                                    np.sort(next_partition))
        if not consistent:
            inconsistent_partitions.append((current_partition.tolist(),
                                                next_partition.tolist()))
        current_partition = statistics_df.loc[i, 'partition']


    count_unique = len(unique_partitions)
    if count_unique > 1:
        print(f"Found {count_unique} unique partitions in block {block_id}:")
        for partition in unique_partitions:
            print(f"\t{partition}")
        print(f"\tCore analysis: {remove_common_head_and_tail(unique_partitions)}")


def build_statistics_df(N, record_observations=False):
    # Check if N exceeds the computational limit to decide on the generation method
    if N > COMP_LIMIT:
        # Use random sampling for large N
        sequences = [''.join(np.random.choice(['0', '1'], N)) for _ in range(2**COMP_LIMIT)]
    else:
        # Generate all possible binary sequences for N <= 20
        sequences = [''.join(seq) for seq in product('01', repeat=N)]

    # Prepare to store chi-squared values
    observations = []
    chi_squared_values = []

    # Get expected counts
    expected_counts = get_expectations(N)

    # Calculate chi-squared for each sequence
    for seq in sequences:
        # run_counts = get_run_counts(seq)
        observed_counts, observed_partition = get_observations(seq)
        chi_squared = get_chi_squared(observed_counts, expected_counts)
        chi_squared_values.append((seq, chi_squared))  
        if record_observations:
            # run_partition = get_run_partition(seq)
            observations.append((seq, observed_counts, observed_partition))

    # Create a DataFrame from the results
    statistics_df = pd.DataFrame(chi_squared_values, columns=['key', 'chi_squared'])

    # Add observations to dataframe if requested
    if record_observations:
        observations_df = pd.DataFrame(observations, columns=['key', 'counts', 'partition'])
        statistics_df = pd.merge(observations_df, statistics_df, on='key', how='left')

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
            # End of block
            block_end_index = next_index

            # Analyze partitions within block
            if record_observations:
                analyze_block(statistics_df, block_start_index, block_end_index)

            # Calculate p-value for the block
            p_value = (block_end_index) / total
            p_values[block_start_index : block_end_index] = p_value

            # Move the start of the next block to the next element
            block_start_index  = next_index  


    statistics_df['p_value'] = p_values

    return statistics_df


def get_db_path():
    # Ensure the folder_path directory exists
    os.makedirs(DB_FOLDER_PATH, exist_ok=True)
    # Returned the full database path
    return os.path.join(DB_FOLDER_PATH, DB_FILE_NAME)


def build_database(lower_bound, upper_bound, record_observations=False, verbose=False):
    # Initialize or open the HDF5 databse
    with pd.HDFStore(get_db_path(), 'a') as store:  # 'a' for read/write if it exists, create otherwise
        # Determine which dataframes already exist
        existing_keys = set(store.keys())

        # Build and add the dataframes
        for N in range(lower_bound, upper_bound + 1):
            key = f'statistics/length_{N}'
            found_key = key in existing_keys

            if not found_key:
                statistics_df = build_statistics_df(N, record_observations)
                store.put(key, statistics_df, format='table', data_columns=True)
            if verbose:
                print(f"\tStatistics for length_{N} {'found' if found_key else 'generated'}.")    


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
