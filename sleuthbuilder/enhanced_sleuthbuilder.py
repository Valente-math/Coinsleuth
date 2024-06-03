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
def set_db_folder_path(folder_path):
    global DB_FOLDER_PATH
    DB_FOLDER_PATH = folder_path

DB_FILE_NAME = 'ultimate_database.h5'
def set_db_file_name(file_name):
    global DB_FILE_NAME
    DB_FILE_NAME = file_name

CHI_SQUARED = 'chi_squared'
LOG_CHI_SQUARED = 'log_chi_squared'
P_VALUE = 'p_value'
# TODO: MAX_RUN = 'max_run'
#       AVG_RUN = 'avg_run'
#       BALANCE = 'balance'
#       P_STAR  = 'p_star'
TEST_STATISTICS = [CHI_SQUARED, LOG_CHI_SQUARED, P_VALUE]

MEAN = 'mean'
STD_DEV = 'std_dev'
MIN = 'min'
Q1 = 'Q1'
MEDIAN = 'median'
Q3 = 'Q3'
MAX = 'max'
MODE = 'mode'
SUMMARY_STATISTICS = [MEAN, STD_DEV, MIN, Q1, MEDIAN, Q3, MAX, MODE]

USE_DICTS = True
STATISTICS_DICT = {}
SUMMARY_DICT = {}
def set_use_dict(use_dict):
    global USE_DICTS
    USE_DICTS = use_dict



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
        data.append((partition_id, multiplicity, chi_squared))

    statistics_df = pd.DataFrame(data, columns=['partition', 'multiplicity', CHI_SQUARED])
    statistics_df.set_index('partition', inplace=True)  # Set partition as the index
    statistics_df.sort_values(by=CHI_SQUARED, inplace=True)
    
    statistics_df[LOG_CHI_SQUARED] = np.log10(statistics_df[CHI_SQUARED])

    total_multiplicity = statistics_df['multiplicity'].sum()
    # print(f'Total multiplicity for N = {N}: {total_multiplicity} ~ {2**N}') # How did copilot know this? 🤔 
    statistics_df[P_VALUE] = statistics_df['multiplicity'][::-1].cumsum()[::-1] / total_multiplicity


    return statistics_df



### Build Database

def get_db_path():
    # Ensure the folder_path directory exists
    os.makedirs(DB_FOLDER_PATH, exist_ok=True)
    # Returned the full database path
    return os.path.join(DB_FOLDER_PATH, DB_FILE_NAME)

    
def get_db_key(N, name):
    return f'/N_{N}/{name}'


def extract_seq_length(key):
    return int(key.split('_')[1].split('/')[0])


def record_data(store, key, data):
    store.put(key, data, format='table', data_columns=True)


def build_database(lower_bound, upper_bound):
    db_path = get_db_path()  # Get the path to the database

    with pd.HDFStore(db_path, mode='a') as store:  # Open the store in append mode
        for N in range(lower_bound, upper_bound + 1):
            db_key = get_db_key(N, 'statistics')
            if db_key not in store.keys():  
                statistics_df = calculate_statistics(N)
                record_data(store, db_key, statistics_df)
        return store


def get_statistics(N):
    # Attempt to retrieve statistics from memory, if available. Otherwise, calculate and return.
    if N in STATISTICS_DICT:
        # Pull from the dictionary
        return STATISTICS_DICT[N]
    else:
        if USE_DB:
            db_path = get_db_path()
            db_key = get_db_key(N, 'statistics')
            with pd.HDFStore(db_path, mode='a') as store:
                if db_key not in store.keys():
                    build_database(N, N)
                statistics_df = store[db_key]
        else:
            statistics_df = calculate_statistics(N)
        if USE_DICTS:
            STATISTICS_DICT[N] = statistics_df
        return statistics_df


def calculate_summary(N):
    statistics_df = get_statistics(N)
    multiplicities = statistics_df['multiplicity'].values
    cumulative_multiplicities = np.cumsum(multiplicities)
    total = 2**N

    summary_data = [] 

    for test_stat in TEST_STATISTICS:
        # Extract statistic values
        statistic_values = statistics_df[test_stat].values
        
        summary_statistics = {} 

        # Calculate the mean
        if MEAN in SUMMARY_STATISTICS:
            mean_val = np.sum(statistic_values * multiplicities) / total
            summary_statistics[MEAN] = mean_val

        # Calculate the standard deviation
        if STD_DEV in SUMMARY_STATISTICS:
            std_dev = np.sqrt(np.sum(((statistic_values - mean_val)**2) * multiplicities) / (total - 1))
            summary_statistics[STD_DEV] = std_dev
        
        # Calculate the mode
        if MODE in SUMMARY_STATISTICS:
            mode_index = np.argmax(multiplicities)
            mode_val = statistic_values[mode_index]
            summary_statistics[MODE] = mode_val
        
        # Calculate the median and quartiles
        if Q1 in SUMMARY_STATISTICS:
            q1_val = statistic_values[np.searchsorted(cumulative_multiplicities, 0.25 * total)]
            summary_statistics[Q1] = q1_val

        if MEDIAN in SUMMARY_STATISTICS:
            median_val = statistic_values[np.searchsorted(cumulative_multiplicities, 0.50 * total)]
            summary_statistics[MEDIAN] = median_val

        if Q3 in SUMMARY_STATISTICS:
            q3_val = statistic_values[np.searchsorted(cumulative_multiplicities, 0.75 * total)]
            summary_statistics[Q3] = q3_val
        
        # Calculate the min and max
        if MIN in SUMMARY_STATISTICS:
            min_val = np.min(statistic_values)
            summary_statistics[MIN] = min_val

        if MAX in SUMMARY_STATISTICS:
            max_val = np.max(statistic_values)
            summary_statistics[MAX] = max_val

        summary = [summary_statistics[stat] for stat in SUMMARY_STATISTICS]
        summary_data.append([test_stat, *summary])

    summary_df = pd.DataFrame(summary_data, columns=['test_stat',*SUMMARY_STATISTICS])
    summary_df.set_index('test_stat', inplace=True)
    return summary_df


def summarize_database():
    db_path = get_db_path()  

    with pd.HDFStore(db_path, mode='a') as store: 
        for key in store.keys():
            if 'statistics' in key:
                N = extract_seq_length(key)
                summary_key = get_db_key(N, 'summary')
                summary_df = calculate_summary(N)
                record_data(store, summary_key, summary_df)


def get_summary(N):
    # Attempt to retrieve from memory, if available. Otherwise, calculate and return.
    if N in SUMMARY_DICT:
        # Pull from the dictionary
        return SUMMARY_DICT[N]
    else:
        if USE_DB:
            db_path = get_db_path()
            summ_key = get_db_key(N, 'summary')
            with pd.HDFStore(db_path, mode='a') as store:
                if summ_key not in store.keys():
                    stat_key = get_db_key(N, 'statistics')
                    if stat_key not in store.keys():
                        build_database(N, N)
                    summarize_database()
                summary_df = store[summ_key]
        else:
            summary_df = calculate_summary(N)
        if USE_DICTS:
            SUMMARY_DICT[N] = summary_df
        return summary_df


def load_database():
    if not USE_DB or not USE_DICTS:
        return
    # Load the database into memory
    db_path = get_db_path()
    with pd.HDFStore(db_path, mode='r') as store:
        for key in store.keys():
            N = extract_seq_length(key)
            if 'statistics' in key:
                STATISTICS_DICT[N] = store[key]
            if 'summary' in key:
                SUMMARY_DICT[N] = store[key]
    print('Database loaded into memory.')


print("Ultimate Sleuthbuilder ready!")