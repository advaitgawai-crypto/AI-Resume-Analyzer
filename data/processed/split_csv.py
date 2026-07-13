import csv
import sys

def split_csv(input_file, output_dir='split_columns'):
    """
    Split a CSV file into 7 separate text files (one per column).
    Each row becomes a new line in its respective column file.
    
    Usage:
        python split_csv.py <input_csv_file> [output_directory]
    
    Output files:
        split_columns/column_0.txt (or 1, 2, 3, 4, 5, 6)
    """
    
    import os
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created directory: {output_dir}")
    
    # Open all 7 output files
    output_files = []
    try:
        for i in range(7):
            filename = os.path.join(output_dir, f'column_{i}.txt')
            output_files.append(open(filename, 'w', encoding='utf-8'))
        
        # Read CSV and split
        with open(input_file, 'r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            row_count = 0
            
            for row in reader:
                # Ensure row has exactly 7 columns
                if len(row) != 7:
                    print(f"⚠ Row {row_count + 1} has {len(row)} columns, expected 7. Skipping.")
                    continue
                
                # Write each column value to its corresponding file
                for col_idx, value in enumerate(row):
                    output_files[col_idx].write(value + '\n')
                
                row_count += 1
            
            print(f"✓ Split {row_count} rows into 7 files")
    
    finally:
        # Close all files
        for f in output_files:
            f.close()
    
    print(f"✓ Output files created in: {output_dir}/")
    for i in range(7):
        filename = os.path.join(output_dir, f'column_{i}.txt')
        print(f"  - {filename}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python split_csv.py <input_csv_file> [output_directory]")
        print("Example: python split_csv.py cleaned500.csv split_columns")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'split_columns'
    
    split_csv(input_file, output_dir)
