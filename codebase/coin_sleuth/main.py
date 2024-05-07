import pandas as pd

def get_p_value_for_string(binary_string):
    # Assume the HDF5 file is stored in the 'data' folder
    file_path = 'data/statistics_database.h5'
    
    # Determine the length of the binary string to find the corresponding dataset
    length = len(binary_string)
    key = f'/statistics/length_{length}'

    # Open the HDF5 file and read the specific dataset
    with pd.HDFStore(file_path, 'r') as store:
        if key in store:
            df = store[key]
            # Check if the binary string is in the DataFrame
            result = df[df['key'] == binary_string]
            if not result.empty:
                return result['p_value'].values[0]
            else:
                return "String not found in the database."
        else:
            return "No data available for strings of this length."

# Prompt the user for a binary string
user_input = input("Please enter a binary string: ")

# Ensure the input is a valid binary string
if all(c in '01' for c in user_input):
    p_value = get_p_value_for_string(user_input)
    print("P-value:", p_value)
else:
    print("Invalid input. Please make sure to enter a binary string containing only 0s and 1s.")
