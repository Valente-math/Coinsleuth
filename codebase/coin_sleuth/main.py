import sleuthbuilder as sb

# Database parameters
comp_limit = 24
lower_bound = 1
upper_bound = 24
db_file_name = 'exp_database_24.h5'
# db_file_name =f'statistics_database_({lower},{upper})_{comp_limit}.h5'

# Set computational limit
sb.set_comp_limit(comp_limit)

# Set database path
sb.set_db_file_name(db_file_name)

# Build database
sb.build_statistics_database(lower_bound, upper_bound, 
                             store_observations=False, 
                             store_expectations=True,
                             store_statistics=False,
                             verbose=True)