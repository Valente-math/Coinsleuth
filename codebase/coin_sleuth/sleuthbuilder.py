import os
from itertools import product

import numpy as np
import pandas as pd

# Conventions:
# - A 'sequence' is a binary string that represents a sequence of coin flips.
# - The variable 'N' reprents the length of the sequences under consideration.

COMP_LIMIT = 10
def set_comp_limit(depth):
    global COMP_LIMIT
    COMP_LIMIT = depth


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


def get_chi_squared(observed, expected, verbose=False):
    components = (observed - expected)**2 / expected
    chi_squared = np.sum(components)
    if verbose:
        print(f'Observed:\n{observed}\n')
        print(f'Expected:\n{expected}\n')
        print(f'Components:\n{components}\n-> {chi_squared}')
    return chi_squared


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

    # Expected run count floor value:
    # This would only be used if N > COMP_LIMIT and observations are sampled
    MIN_EXPECTATION = (0.5)**(N)

    # Calculate the expected counts for each run length up to 'max_length'
    expected_counts = []
    for run_length in range(1, N + 1):
        # Calculate the mean count for the current run length
        mean_count = max(observations_df[f'runs_{run_length}'].mean(), MIN_EXPECTATION)
        expected_counts.append(mean_count)

    # Create a new DataFrame for the expectation data
    # (there will be only one row in this DataFrame)
    columns = ['length'] + [f'exp_runs_{i+1}' for i in range(N)]
    expectations_df = pd.DataFrame([[N] + expected_counts], columns=columns)

    return expectations_df


def build_statistics_df(observations_df, expectations_df):
    # Extract expected counts from the expectations_df
    expected_counts = expectations_df.iloc[0, 1:].values.astype(float)

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

    # Calculate p-values using the method described
    p_values = np.zeros(len(statistics_df))
    j = 0  # Flag to mark the start of a new block of distinct chi-squared values
    total = len(statistics_df)  # Total number of observations

    # Iterate over the sorted DataFrame
    for i in range(total):
        if i == total - 1 or statistics_df.loc[i, 'chi_squared'] > statistics_df.loc[i + 1, 'chi_squared']:
            # Calculate p-value for the block
            p_value = (i + 1) / total
            p_values[j:i+1] = p_value
            j = i + 1  # Move the start of the next block to the next element

    statistics_df['p_value'] = p_values

    return statistics_df


################################ BEGIN USER-FACING METHODS ################################


def build_statistics_database(lower, upper,
                              filename='statistics_database.h5',
                              store_observations=False,
                              store_expectations=True,
                              verbose=False):
    # Define the path to the HDF5 file within the 'data' folder
    folder_path = 'data'
    file_path = os.path.join(folder_path, filename)

    # Ensure the 'data' directory exists
    os.makedirs(folder_path, exist_ok=True)

    # Initialize or open the HDF5 file
    with pd.HDFStore(file_path, 'a') as store:  # 'a' for read/write if it exists, create otherwise

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


def get_statistics_for_sequence(sequence, db_file_path='data/statistics_database.h5', verbose=False):
    N = len(sequence)
    id = f'length_{N}'
    statistics_key = f'/statistics/{id}'
    expectations_key = f'/expectations/{id}'

    with pd.HDFStore(db_file_path, 'a') as store:  # 'a' to allow data addition if necessary
        # Check and load expectations data
        if expectations_key in store:
            if verbose:
                print(f"Found expected counts for {id}.")
            expectations_df = store[expectations_key]
        else:
            # Prompt for generating expectations data
            if verbose:
                print(f"Expectations data for {id} not found. Generating now...")
            observations_df = build_observations_df(N)
            expectations_df = build_expectations_df(observations_df)
            if input("Save new expectations data to database? (y/n): ") == 'y':
                store.put(expectations_key, expectations_df, format='table', data_columns=True)

        # Compute chi-squared statistic
        observed = get_run_counts(sequence)
        expected = expectations_df.iloc[0, 1:].values.astype(float)
        chi_squared = get_chi_squared(observed, expected)

        # Check and load statistics data
        if statistics_key in store:
            if verbose:
                print(f"Found statistics for {id}.")
            statistics_df = store[statistics_key]
        else:
            # Prompt for generating statistics data
            if verbose:
                print(f"Statistics data for {id} not found. Generating now...")
            observations_df = build_observations_df(N)
            statistics_df = build_statistics_df(observations_df, expectations_df)
            if input("Save new statistics data to database? (y/n): ") == 'y':
                store.put(statistics_key, statistics_df, format='table', data_columns=True)

        # Find p-value or range
        p_value = statistics_df['p_value'][statistics_df['chi_squared'] == chi_squared]
        if verbose:
            print(f"Search results for p-value:\n{p_value}")

        if p_value.empty:
            lower_p_value = statistics_df['p_value'][statistics_df['chi_squared'] > chi_squared]  
            if verbose:
                print(f"Search results for lower p-value:\n{lower_p_value}")

            upper_p_value = statistics_df['p_value'][statistics_df['chi_squared'] < chi_squared]  
            if verbose:
                print(f"Search results for upper p-value:\n{upper_p_value}")

            if lower_p_value.empty:
                if upper_p_value.empty:
                    return f"Chi-squared: {chi_squared}, p-value not determined."
                else:
                    return f"Chi-squared: {chi_squared}, p < {upper_p_value.min()}."
            else:
                if upper_p_value.empty:
                    return f"Chi-squared: {chi_squared}, p > {lower_p_value.max()}."
                else:
                    return f"Chi-squared: {chi_squared}, {lower_p_value.max()} < p < {upper_p_value.min()}"
        else:
            return f"Chi-squared: {chi_squared}, p = {p_value.iloc[0]}."


print("Builder ready...\n")
