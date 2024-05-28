import ultimate_sleuthbuilder as usb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import concurrent.futures

USE_MULTITHREADING = True
def set_multithreading(multithread):
    global USE_MULTITHREADING
    USE_MULTITHREADING = multithread

def multithreading_enabled():
    print(f'Multithreading enabled with {os.cpu_count()} cores available.')
    
def initialize_sleuthbuilder(use_db=usb.USE_DB,
                             use_dict=usb.USE_DICT,
                             db_folder_path=usb.DB_FOLDER_PATH,
                             db_file_name=usb.DB_FILE_NAME):
    usb.set_use_db(use_db)
    usb.set_use_dict(use_dict)
    usb.set_db_folder_path(db_folder_path)
    usb.set_db_file_name(db_file_name)
    usb.load_database()


def get_sequence_partition_id(seq):
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
    return usb.get_partition_id(np.array(partition))


def analyze_sequence(seq):
    N = len(seq)
    results = {'sequence' : seq, 'length' : N}

    statistics_df = usb.get_statistics(N)

    partition_id = get_sequence_partition_id(seq)
    for stat in usb.TEST_STATISTICS:
        results[stat] = statistics_df.loc[partition_id, stat]

    return results


def analyze_sequence_set(sequences):
    if USE_MULTITHREADING:
        multithreading_enabled()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(analyze_sequence, sequences))
    else:
        results = []
        for seq in sequences:
            results.append(analyze_sequence(seq))

    # Return results as dataframe
    return pd.DataFrame(results)


def analyze_sequences_from_csv(filename):
    # Read in data from CSV file
    data = pd.read_csv(filename)
    # Filter out by 'type' column
    data = data[data['type'] == 'initial']
    sequences = data['sequence']
    analysis = analyze_sequence_set(sequences)
    # Convert sequences to numpy array
    # sequences = np.array(sequences)
    # Add analysis to dataframe
    for stat in usb.TEST_STATISTICS:
        data[stat] = analysis[stat]
    return data
    # return analyze_sequence_set(sequences)


def generate_sample_df(sample_size, N):
    sequences = [''.join(np.random.choice(['0', '1'], N)) for _ in range(sample_size)]
    return analyze_sequence_set(sequences)  


def perform_trial(args):
    sample_size, N, statistic, statistics_df = args
    sequences = [''.join(np.random.choice(['0', '1'], N)) for _ in range(sample_size)]
    statistics = []
    for seq in sequences:
        sample_stats = analyze_sequence(seq, statistics_df)
        statistics.append(sample_stats[statistic])
    # sample_df = usb.analyze_sequence_set(sequences)
    # return sample_df['p_value'].mean()
    return np.mean(statistics)

# TODO: Clean up this function
def build_sampling_distribution(trials, sample_size, N, statistic):
    statistics_df = usb.get_statistics(N)
    
    # Function to run all trials
    def run_trials(trials, sample_size, N, statistic, statistics_df):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            sample_means = list(executor.map(perform_trial, [(sample_size, N, statistic, statistics_df) for _ in range(trials)]))
        return sample_means
    
    if USE_MULTITHREADING:
        multithreading_enabled()
        sample_means = run_trials(trials, sample_size, N, statistic, statistics_df)
    else:
        sample_means = []
        for trial in range(trials):
            args = (sample_size, N, statistic, statistics_df)
            sample_mean = perform_trial(args)
            sample_means.append(sample_mean)

    sampling_distribution = pd.DataFrame(sample_means, columns=[f'Mean {statistic}'])
    return sampling_distribution


def calculate_moes(N, sample_size, statistic):
    db_path = usb.get_db_path()  # Get the path to the database
    key = usb.get_db_key(f'summary/{statistic}')

    z_scores = {0.90 : 1.645, 0.95 : 1.960, 0.99 : 2.576, 0.999 : 3.291}
    confidence_levels = z_scores.keys()
    moes = {level : [] for level in confidence_levels}
    for level in confidence_levels:
        # Open the hd5 database at usb.get_db_path()

        with pd.HDFStore(db_path, mode='r') as store:  # Open the store in append mode
            summary_df = store[key]
            
            # Get standard deviation for sequences of length N
            std_dev = summary_df.loc[N, 'std_dev']
            margin_of_error = z_scores[level] * std_dev / np.sqrt(sample_size)
            moes[level].append(margin_of_error)

    return moes


# TODO: Visualization
def plot_distribution(N, statistic):
    return



