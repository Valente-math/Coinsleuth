import ultimate_sleuthbuilder as usb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import concurrent.futures

def load_database():
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
    for stat in usb.STATISTICS:
        results[stat] = statistics_df.loc[partition_id, stat]

    return results


def analyze_sequence_set(sequences):    
    # Initialize results dataframe
    results = []
    
    for seq in sequences: 
        N = len(seq)
        sequence_stats = analyze_sequence(seq)
        # Add sequence_stats as new row of results dataframe
        results.append(sequence_stats)

    # Return results as dataframe
    return pd.DataFrame(results)


def analyze_sequences_from_csv(data):
    # Read in data from CSV file
    data = pd.read_csv(data)
    sequences = data['sequence']
    # Convert sequences to numpy array
    # sequences = np.array(sequences)
    return analyze_sequence_set(sequences)


def generate_sample_df(sample_size, N):
    sequences = [''.join(np.random.choice(['0', '1'], N)) for _ in range(sample_size)]
    # Analyze sequence set and store in dataframe
    sample_df = pd.DataFrame(analyze_sequence_set(sequences))
    return sample_df  



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


def build_sampling_distribution(trials, sample_size, N, statistic,
                                use_db=True,
                                multithread=True):
    usb.set_use_db(use_db)
    statistics_df = usb.get_statistics(N)

    
    # Function to run all trials
    def run_trials(trials, sample_size, N, statistic, statistics_df):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            sample_means = list(executor.map(perform_trial, [(sample_size, N, statistic, statistics_df) for _ in range(trials)]))
        return sample_means
    
    if multithread:
        print(f'Number of cores available: {os.cpu_count()}')
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

    # print('Margin of Error for Various Confidence Levels')
    # for level in confidence_levels:
    #     print(f'{level} confidence level: {moes[level]}')

    return moes


def plot_population_distribution(N, statistic, num_bins=20):
    statistics_df = usb.get_statistics(N)

    # Sample data creation
    data = {'x': statistics_df[statistic].values,
            'y': statistics_df['multiplicity'].values}

    df = pd.DataFrame(data)

    # Create the bins
    if statistic == 'p_value':
        bins = np.linspace(0, 1, num_bins + 1)
    else:
        bins = np.linspace(df['x'].min(), df['x'].max(), num_bins + 1)

    # Bin the data and aggregate the counts
    df['bin'] = pd.cut(df['x'], bins=bins, labels=False, include_lowest=True)
    binned_data = df.groupby('bin')['y'].sum().reset_index()

    # Convert bin numbers back to bin ranges for plotting
    binned_data['bin'] = bins[:-1] + (bins[1] - bins[0]) / 2

    return binned_data

    # Plotting the histogram
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x='bin', y='y', data=binned_data, palette='viridis')
    # plt.xlabel('Values')
    # plt.ylabel('Counts')
    # plt.title('Binned Histogram of Values vs Counts')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.show()



