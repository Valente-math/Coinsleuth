import timeit
import sleuthbuilder as sb
import pandas as pd

def test_algorithms(depth, trials=10):
    # Set up
    observations_df = sb.build_observations_df(depth)
    expectations_df = sb.build_expectations_df(observations_df)

    # Functions for timeit
    global test_v0, test_v1, test_v2

    def test_v0():
        sb.build_expectations_df(sb.build_observations_df(depth))

    def test_v1():
        sb.build_statistics_df_v1(observations_df, expectations_df)

    def test_v2():
        sb.build_statistics_df_v2(observations_df, expectations_df)

    # Timing each function using timeit
    setup_time = timeit.timeit('test_v0()', globals=globals(), number=trials)/trials
    print(f"Average setup time at depth {depth} over {trials} runs: {setup_time} seconds")

    time_v1 = timeit.timeit('test_v1()', globals=globals(), number=trials)/trials
    print(f"Average time taken by v1 at depth {depth} over {trials} runs: {time_v1} seconds")

    time_v2 = timeit.timeit('test_v2()', globals=globals(), number=trials)/trials
    print(f"Average time taken by v2 at depth {depth} over {trials} runs: {time_v2} seconds")

    print()

    return (setup_time, time_v1, time_v2)


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

results_df.to_csv('speedtest.csv', index=False)


