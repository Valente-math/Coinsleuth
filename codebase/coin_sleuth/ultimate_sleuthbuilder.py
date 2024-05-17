import os
import numpy as np
import pandas as pd
from math import factorial



### Globals

DB_FOLDER_PATH = 'data'
def set_db__folder_path(folder_path):
    global DB_FOLDER_PATH
    DB_FOLDER_PATH = folder_path


DB_FILE_NAME = 'ultimate_database.h5'
def set_db_file_name(file_name):
    global DB_FILE_NAME
    DB_FILE_NAME = file_name



### Calculate Statistics

def get_expected_counts(N):
    # Define coefficient function
    def c(n, N):
        if n >  N:
            return 0
        if n == N:
            return 1
        if n <  N:
            return 0.25*(3 + N - n)

    # Calculate probability function
    expected_counts = [c(n, N)*(2**(1-n)) for n in range(1, N + 1)]
    return np.array(expected_counts)


def integer_partitions(n):
    a = [0] * (n + 1)
    k = 1
    y = n - 1
    a[0] = 0
    a[1] = n
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]


def get_chi_squared(observed, expected):
    components = (observed - expected)**2 / expected
    return np.sum(components)


def count_multiplicity(partition):
    # Count occurrences of each number in the partition using a simple dictionary
    count_dict = {}
    for num in partition:
        if num in count_dict:
            count_dict[num] += 1
        else:
            count_dict[num] = 1
    
    # Calculate the factorial of the length of the partition
    total_length_factorial = factorial(len(partition))
    
    # Calculate the product of the factorials of the counts
    denominator = 1
    for count in count_dict.values():
        denominator *= factorial(count)
    
    # Calculate multiplicity
    multiplicity = total_length_factorial // denominator
    
    # Adjust for symmetry
    multiplicity *= 2
    
    return multiplicity


def calculate_statistics(N):
    expected_counts = get_expected_counts(N)
    data = []

    for partition in integer_partitions(N):
        observed_counts = np.zeros(N, dtype=int)
        for k in partition:
            observed_counts[k - 1] += 1  # Adjust index since partitions start at 1

        chi_squared = get_chi_squared(observed_counts, expected_counts)
        multiplicity = count_multiplicity(partition)
        
        partition_str = ','.join(map(str, sorted(partition)))  # Convert partition to a sorted string for storage
        data.append((partition_str, chi_squared, multiplicity))

    statistics_df = pd.DataFrame(data, columns=['partition', 'chi_squared', 'multiplicity'])
    statistics_df.set_index('partition', inplace=True)  # Set partition as the index
    statistics_df.sort_values(by='chi_squared', inplace=True)
    
    total_multiplicity = statistics_df['multiplicity'].sum()
    statistics_df['p_value'] = statistics_df['multiplicity'][::-1].cumsum()[::-1] / total_multiplicity

    return statistics_df



### Build Database

def get_db_path():
    # Ensure the folder_path directory exists
    os.makedirs(DB_FOLDER_PATH, exist_ok=True)
    # Returned the full database path
    return os.path.join(DB_FOLDER_PATH, DB_FILE_NAME)


def get_db_key(name, N=None):
    if N is None:
        return f'/{name}'
    else:
        return f'/{name}/N_{N}'


def record_data(store, key, data):
    store.put(key, data, format='table', data_columns=True)


def summarize_database():
    db_path = get_db_path()  # Get the path to the database
    summary = []

    with pd.HDFStore(db_path, mode='a') as store:  # Open the store in read mode
        for key in store.keys():
            N = int(key.split('_')[1])
            statistics_df = store[key]
            
            # Extract chi-squared values and their multiplicities
            chi_squared_values = statistics_df['chi_squared'].values
            multiplicities = statistics_df['multiplicity'].values
            
            # Calculate the total multiplicity
            total_multiplicity = np.sum(multiplicities)
            
            # Calculate the mean
            mean_val = np.sum(chi_squared_values * multiplicities) / total_multiplicity
            
            # Calculate the mode
            mode_index = np.argmax(multiplicities)
            mode_val = chi_squared_values[mode_index]
            
            # Calculate the five-number summary
            cumulative_multiplicities = np.cumsum(multiplicities)
            min_val = chi_squared_values[0]
            Q1 = chi_squared_values[np.searchsorted(cumulative_multiplicities,
                                                     0.25 * total_multiplicity)]
            Q2 = chi_squared_values[np.searchsorted(cumulative_multiplicities, 
                                                    0.50 * total_multiplicity)]  # Median
            Q3 = chi_squared_values[np.searchsorted(cumulative_multiplicities, 
                                                    0.75 * total_multiplicity)]
            max_val = chi_squared_values[-1]
            
            # Append the summary statistics for this N
            summary.append([N, min_val, Q1, Q2, Q3, max_val, mean_val, mode_val])
    
        # Create a DataFrame from the summary
        summary_df = pd.DataFrame(summary, columns=['N', 'min', 'Q1', 'Q2', 'Q3', 'max', 'mean', 'mode'])
        
        # Sort by N and set as the index
        summary_df.sort_values(by='N', inplace=True)
        summary_df.set_index('N', inplace=True)

        # Save to database
        summary_key = get_db_key('summary')
        record_data(store, summary_key, summary_df)

        return summary_df


def build_database(lower_bound, upper_bound, summarize=False):
    db_path = get_db_path()  # Get the path to the database

    with pd.HDFStore(db_path, mode='a') as store:  # Open the store in append mode
        for N in range(lower_bound, upper_bound + 1):
            db_key = get_db_key('statistics', N)
            if db_key in store.keys():  # Check if the key already exists in the database
                print(f'Skipping N = {N}, already exists in the database.')
            else:
                print(f'Processing N = {N}...')
                statistics_df = calculate_statistics(N)
                record_data(store, db_key, statistics_df)
        if summarize:
            summarize_database()
        return store



### Analyze Sequences

def get_sequence_partition(seq):
    N = len(seq)
    partition = []
    current_run_length = 1
    current_char = seq[0]

    for i in range(1, N):
        if seq[i] == current_char:
            # Run continues
            current_run_length += 1
        else:
            # Run ended by opposing flip
            partition.append(current_run_length)

            # Reset run
            current_run_length = 1
            current_char = seq[i]

    # Run ended by sequence termination
    partition.append(current_run_length)

    return np.array(partition)


def analyze_sequence(seq, use_database=True):
    N = len(seq)
    partition = get_sequence_partition(seq)
    partition.sort()  # Sort the partition in ascending order
    partition_str = ','.join(map(str, partition))  # Convert partition to a string for comparison

    if use_database:
        db_path = get_db_path()  # Get the path to the database
        db_key = get_db_key('statistics', N)
        with pd.HDFStore(db_path, mode='a') as store:
            # print(f'Search for {key} in {store.keys()}')
            if db_key not in store.keys():
                # print(f'Data for N = {N} not found. Generating and saving now...')
                build_database(N, N)
            statistics_df = store[db_key]
    else:
        statistics_df = calculate_statistics(N)

    chi_squared = statistics_df.loc[partition_str, 'chi_squared']
    p_value = statistics_df.loc[partition_str, 'p_value']
    return {'N' : N, 'chi_squared' : chi_squared, 'p_value' : p_value}
            

