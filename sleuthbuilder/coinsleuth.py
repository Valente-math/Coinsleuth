import ultimate_sleuthbuilder as usb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

    
def initialize_sleuthbuilder(use_db=usb.USE_DB,
                             use_dict=usb.USE_DICTS,
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


def analyze_sequence_sample(sequences):
    sample_analysis = []
    for seq in sequences:
        sample_analysis.append(analyze_sequence(seq))

    # Return results as dataframes
    return pd.DataFrame(sample_analysis)


def test_sample(sample_analysis):
    N = len(sample_analysis['sequence'][0]) # Assume all sequences are the same length
    test_results = []
    for stat in usb.TEST_STATISTICS:
        values = sample_analysis[stat]

        sample_mean = np.mean(values)
        pop_mean = usb.get_summary(N).loc[stat, 'mean']
        std_dev = usb.get_summary(N).loc[stat, 'std_dev']
        std_error = std_dev / np.sqrt(len(sample_analysis)) # Population standard deviation / sqrt(sample size)
        z_score = (sample_mean - pop_mean) / std_error
        p_value = stats.norm.sf(abs(z_score)) * 2 # Two-tailed test

        test_results.append((stat, p_value))

    test_results = pd.DataFrame(test_results, columns=['test_stat', 'p_value'])

    # Return results as dataframes
    return test_results


def analyze_sequences_from_csv(filename):
    # Read in data from CSV file
    data = pd.read_csv(filename)
    sequences = data['sequence']
    sample_analysis = analyze_sequence_sample(sequences)
    # Return merged analysis with data
    return pd.merge(data, sample_analysis, on='sequence')


def generate_sample_df(sample_size, N):
    sequences = [''.join(np.random.choice(['0', '1'], N)) for _ in range(sample_size)]
    return analyze_sequence_sample(sequences)  


# TODO: Visualization
# def plot_distribution(N, statistic):
#     return



