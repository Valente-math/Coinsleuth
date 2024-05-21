import timeit
import numpy as np
import pandas as pd
import sleuthbuilder as sb

def build_statistics_df_v1(observations_df, expectations_df):
    # Extract expected counts from the expectations_df
    expected_counts = expectations_df.iloc[0, 1:].values.astype(float)

    # Prepare to store 𝜒^2 values
    chi_squared_results = []

    # Iterate over each row in observations_df to calculate 𝜒^2 values
    for index, row in observations_df.iterrows():
        observed_counts = row[1:].values.astype(float)
        # Calculate the 𝜒^2 statistic
        chi_squared = np.sum((observed_counts - expected_counts)**2 / expected_counts)
        chi_squared_results.append((row['key'], chi_squared))

    # Create a DataFrame from the results
    statistics_df = pd.DataFrame(chi_squared_results,
                                 columns=['key', 'chi_squared'])

    # Calculate p-values as the proportion of all 𝜒^2 values 
    # that are <= each observed 𝜒^2 value
    all_chi_squared_values = statistics_df['chi_squared'].values
    statistics_df['p_value'] = [
        np.mean(all_chi_squared_values >= x) for x in all_chi_squared_values
    ]

    return statistics_df


def build_statistics_df_v2(observations_df, expectations_df):
    # Extract expected counts from the expectations_df
    expected_counts = expectations_df.iloc[0, 1:].values.astype(float)

    # Prepare to store chi-squared values
    chi_squared_results = []

    # Iterate over each row in observations_df to calculate chi-squared values
    for index, row in observations_df.iterrows():
        observed_counts = row[1:].values.astype(float)
        # Calculate the chi-squared statistic
        chi_squared = np.sum((observed_counts - expected_counts)**2 / expected_counts)
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


def build_statistics_df_v3(observations_df, expectations_df):
    # Extract expected counts from the expectations_df
    expected_counts = expectations_df['mean'].values.astype(float)

    # Prepare to store chi-squared values
    chi_squared_results = []

    # Iterate over each row in observations_df to calculate chi-squared values
    for index, row in observations_df.iterrows():
        observed_counts = row[1:].values.astype(float)
        # Calculate the chi-squared statistic
        chi_squared = np.sum((observed_counts - expected_counts)**2 / expected_counts)
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

 
def build_statistics_df_v4(observations_df, expectations_df):
    # Extract expected counts from the expectations_df
    expected_counts = expectations_df['mean'].values.astype(float)

    # Prepare to store chi-squared values
    chi_squared_results = []

    # Iterate over each row in observations_df to calculate chi-squared values
    for index, row in observations_df.iterrows():
        observed_counts = row[1:].values.astype(float)
        # Calculate the chi-squared statistic
        chi_squared = sb.get_chi_squared(observed_counts, expected_counts)
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


def test_algorithms(depth, trials=10):
    # Set up
    observations_df = sb.build_observations_df(depth)
    expectations_df = sb.build_expectations_df(observations_df)

    # Functions for timeit
    global test_v0, test_v1, test_v2

    def test_v0():
        sb.build_expectations_df(sb.build_observations_df(depth))

    def test_v1():
        build_statistics_df_v3(observations_df, expectations_df)

    def test_v2():
        build_statistics_df_v4(observations_df, expectations_df)

    # Timing each function using timeit
    setup_time = timeit.timeit('test_v0()', globals=globals(), number=trials)/trials
    print(f"Average setup time at depth {depth} over {trials} runs: {setup_time} seconds")

    time_v1 = timeit.timeit('test_v1()', globals=globals(), number=trials)/trials
    print(f"Average time taken by v1 at depth {depth} over {trials} runs: {time_v1} seconds")

    time_v2 = timeit.timeit('test_v2()', globals=globals(), number=trials)/trials
    print(f"Average time taken by v2 at depth {depth} over {trials} runs: {time_v2} seconds")

    print()

    return (setup_time, time_v1, time_v2)


sb.set_comp_limit(20)

# Testing at different depths and storing ratios
depths = range(1, 21)  # Depths from 1 to 20
results = [test_algorithms(depth) for depth in depths]

# Creating a DataFrame
results_df = pd.DataFrame({
    'Depth': depths,
    'Setup Time (seconds)': [result[0] for result in results],
    'Time v1 (seconds)': [result[1] for result in results],
    'Time v2 (seconds)': [result[2] for result in results]
})

# Calculate the ratio of the times v1/v2
# If ratio > 1, then v2 is faster
# If ratio < 1, then v1 is faster
results_df['Ratio (v1/v2)'] = results_df['Time v1 (seconds)'] / results_df['Time v2 (seconds)']

results_df.to_csv('speedtest2.csv', index=False)


