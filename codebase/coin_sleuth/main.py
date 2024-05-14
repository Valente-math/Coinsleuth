import sleuthbuilder as sb
# import argparse

# Set database parameters
comp_limit = 10
lower = 1
upper = 20

sb.set_comp_limit(comp_limit)

# Set database path
db_folder_path = 'data'
db_file_name =f'statistics_database_({lower},{upper})_{comp_limit}.h5'
# db_file_name = 'test_database.h5'
sb.set_db_path(db_folder_path, db_file_name)

# Build database
sb.build_statistics_database(lower, upper, 
                             store_observations=False, 
                             store_expectations=True,
                             verbose=False)


# # Prompt the user for a binary string
# user_input = input("Please enter a binary string: ")

# # Ensure the input is a valid binary string
# if all(c in '01' for c in user_input):
#     p_value = sb.get_p_value_for_sequence(user_input)
#     print("P-value:", p_value)
# else:
#     print("Invalid input. Please make sure to enter a binary string containing only 0s and 1s.")


# def run():
#     parser = argparse.ArgumentParser(description='Build a statistics database for binary strings.')
#     parser.add_argument('depth', type=int, help='The depth of statistics to build.')
#     parser.add_argument('--csv', action='store_true', help='Enable CSV export.')
#     parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')

#     args = parser.parse_args()

#     build_statistics_database(args.depth, csv_export=args.csv, verbose=args.verbose)

# if __name__ == '__main__':
#     run()