import argparse
import sleuthbuilder as sb

def run():
    parser = argparse.ArgumentParser(description="CLI for sleuthbuilder methods")
    
    # Add subparsers for each method in sleuthbuilder
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Global optional arguments
    parser.add_argument('--verbose', action='store_true', help='Enables verbose output.')
    parser.add_argument('--limit', type=int, help='Set the computation limit')
    parser.add_argument('--db', type=str, help='Set the database filename.')


    # Add 'build_statistics_database'
    method1_parser = subparsers.add_parser('build', help='build_statistics_database')
    method1_parser.add_argument('lower_bound', type=int, help='Lower bound for database depth.')
    method1_parser.add_argument('upper_bound', type=int, help='Upper bound for database depth.')

    # Grab arguments    
    args = parser.parse_args()

    if args.limit is not None:
        sb.set_comp_limit(args.limit)
        print(f"Computation limit set to {args.limit}")

    if args.db is not None:
        sb.set_db_file_name(args.db)
        print(f"Database set to {args.db}")

    if args.command == 'build':
        sb.build_statistics_database(args.lower_bound, args.upper_bound, verbose=args.verbose)
    

if __name__ == '__main__':
    run()


# Example usage:
# python cli.py --verbose --limit 10 --db test_database.h5  build 1 12