import os
import numpy as np
import pandas as pd
from math import factorial


### Globals

USE_DB = True
def set_use_db(use_db):
    global USE_DB
    USE_DB = use_db

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


def get_partition_id(partition):
    return '+'.join(map(str, sorted(partition)))  
    # return hash(tuple(sorted(partition)))


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
        
        # Convert partition to a sorted string for storage
        partition_id = get_partition_id(partition)
        data.append((partition_id, chi_squared, multiplicity))

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
    statistics = ['chi_squared', 'p_value']

    # Initialize the summary list as dict with statistics as keys
    summary = {stat: [] for stat in statistics}

    with pd.HDFStore(db_path, mode='a') as store:  # Open the store in read mode
        for key in store.keys():
            N = int(key.split('_')[1])
            statistics_df = store[key]

            # Calculate multiplities and cumulative totals
            multiplicities = statistics_df['multiplicity'].values
            cumulative_multiplicities = np.cumsum(multiplicities)
            total = np.sum(multiplicities)

            for stat in statistics:
                # Extract statistic values
                statistic_values = statistics_df[stat].values
                
                # Calculate the mean
                mean_val = np.sum(statistic_values * multiplicities) / total

                # Calculate the standard deviation
                std_dev = np.sqrt(np.sum(((statistic_values - mean_val)**2) * multiplicities) / (total - 1))
                
                # Calculate the mode
                mode_index = np.argmax(multiplicities)
                mode_val = statistic_values[mode_index]
                
                # Calculate the median
                median_val = statistic_values[
                    np.searchsorted(cumulative_multiplicities, 0.50 * total)]
                
                # Append the summary statistics for this N
                summary[stat].append([N, mode_val, median_val, mean_val, std_dev])
    
        for stat in statistics:
            # Create a DataFrame from the summary
            summary_df = pd.DataFrame(summary[stat], columns=['N', 'mode', 'median', 'mean', 'std_dev'])
            
            # Sort by N and set as the index
            summary_df.sort_values(by='N', inplace=True)
            summary_df.set_index('N', inplace=True)

            # Save to database
            summary_key = get_db_key(f'summary/{stat}')
            record_data(store, summary_key, summary_df)

    return summary_df


def build_database(lower_bound, upper_bound, summarize=True):
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


def get_statistics(N):
    if USE_DB:
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
    return statistics_df


def analyze_sequence(seq, statistics_df=None):
    N = len(seq)
    partition = get_sequence_partition(seq)
    partition_id = get_partition_id(partition)

    if statistics_df is None:
        statistics_df = get_statistics(N)

    chi_squared = statistics_df.loc[partition_id, 'chi_squared']
    p_value = statistics_df.loc[partition_id, 'p_value']
    return {
        'sequence' : seq,
        'length' : N, 
        'chi_squared' : chi_squared, 
        'p_value' : p_value
        }
            

def analyze_sequence_set(sequences):
    # Initialize statistics dictionary
    lengths = {len(seq) for seq in sequences}
    statistics = {N : get_statistics(N) for N in lengths}
    
    # Initialize results dataframe
    results = []
    
    for seq in sequences: 
        N = len(seq)
        statistics_df = statistics[N]
        sequence_stats = analyze_sequence(seq, statistics_df)
        # Add sequence_stats as new row of results dataframe
        results.append(sequence_stats)

    # Return results as dataframe
    return pd.DataFrame(results)
    

