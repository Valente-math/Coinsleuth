import os
import numpy as np
import pandas as pd
from math import factorial
from collections import Counter

DB_FOLDER_PATH = 'data'
def set_db__folder_path(folder_path):
    global DB_FOLDER_PATH
    DB_FOLDER_PATH = folder_path

DB_FILE_NAME = 'ultimate_database.h5'
def set_db_file_name(file_name):
    global DB_FILE_NAME
    DB_FILE_NAME = file_name


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
    return np.sum(components)

def partitions(n):
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

def calculate_multiplicity(partition):
    counter = Counter(partition)
    total_count = sum(counter.values())
    multiplicity = factorial(total_count)
    for count in counter.values():
        multiplicity //= factorial(count)
    return multiplicity * 2  # Multiply by 2 to account for symmetry

def analyze_partitions(N):
    expected_counts = get_expectations(N)
    data = []

    for partition in partitions(N):
        observed_counts = np.zeros(N, dtype=int)
        for k in partition:
            observed_counts[k - 1] += 1  # Adjust index since partitions start at 1

        chi_squared = get_chi_squared(observed_counts, expected_counts)
        multiplicity = calculate_multiplicity(partition)
        
        partition_str = ','.join(map(str, sorted(partition)))  # Convert partition to a sorted string for storage
        data.append((partition_str, chi_squared, multiplicity))

    df = pd.DataFrame(data, columns=['partition', 'chi_squared', 'multiplicity'])
    df.set_index('partition', inplace=True)  # Set partition as the index
    df.sort_values(by='chi_squared', inplace=True)
    
    total_multiplicity = df['multiplicity'].sum()
    df['p_value'] = df['multiplicity'][::-1].cumsum()[::-1] / total_multiplicity

    return df

def get_db_path():
    # Ensure the folder_path directory exists
    os.makedirs(DB_FOLDER_PATH, exist_ok=True)
    # Returned the full database path
    return os.path.join(DB_FOLDER_PATH, DB_FILE_NAME)

def get_db_key(N):
    return f'/N_{N}'

def build_database(lower_bound, upper_bound):
    db_path = get_db_path()  # Get the path to the database

    with pd.HDFStore(db_path, mode='a') as store:  # Open the store in append mode
        for N in range(lower_bound, upper_bound + 1):
            key = get_db_key(N)
            if key in store.keys():  # Check if the key already exists in the database
                print(f'Skipping N = {N}, already exists in the database.')
                continue
            print(f'Processing N = {N}...')
            df = analyze_partitions(N)  # Analyze partitions for the current value of N
            store.put(key, df, format='table', data_columns=True)  # Store the DataFrame in the HDF5 database

def get_statistics(sequence):
    N = len(sequence)
    partition = get_partition(sequence)
    partition.sort()  # Sort the partition in ascending order
    partition_str = ','.join(map(str, partition))  # Convert partition to a string for comparison

    db_path = get_db_path()  # Get the path to the database

    key = get_db_key(N)

    with pd.HDFStore(db_path, mode='a') as store:
        # print(f'Search for {key} in {store.keys()}')
        if key not in store.keys():
            print(f'Data for N = {N} not found. Generating and saving now...')
            build_database(N, N)
        df = store[key]
        if partition_str in df.index:
            p_value = df.loc[partition_str, 'p_value']
            return p_value
        else:
            raise KeyError
            

