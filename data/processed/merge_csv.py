import csv
import sys
import os

def merge_csv(input_dir='split_columns', output_file='merged_output.csv', add_header=None):
    """
    Merge 7 separate text files back into a single CSV file.
    Each line in each file becomes a column value for that row.
    
    Usage:
        python merge_csv.py [input_directory] [output_csv_file] [--header "col1,col2,col3,col4,col5,col6,col7"]
    
    Example:
        python merge_csv.py split_columns output.csv
        python merge_csv.py split_columns output.csv --header "ID,SKILL,JOB_TITLE,EXP_DUR,DEGREE,INSTITUTION,CERT"
    """
    
    # Read all 7 column files
    column_data = []
    
    try:
        for i in range(7):
            filename = os.path.join(input_dir, f'column_{i}.txt')
            if not os.path.exists(filename):
                print(f"✗ File not found: {filename}")
                return False
            
            with open(filename, 'r', encoding='utf-8') as f:
                lines = [line.rstrip('\n') for line in f.readlines()]
                column_data.append(lines)
                print(f"✓ Loaded {filename} ({len(lines)} rows)")
        
        # Verify all columns have same number of rows
        num_rows = len(column_data[0])
        for i, col in enumerate(column_data):
            if len(col) != num_rows:
                print(f"✗ Column {i} has {len(col)} rows, expected {num_rows}")
                return False
        
        print(f"✓ All columns have {num_rows} rows")
        
        # Write CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            
            # Write header if provided
            if add_header:
                header = add_header.split(',')
                if len(header) != 7:
                    print(f"⚠ Header has {len(header)} columns, expected 7. Using anyway.")
                writer.writerow(header)
                print(f"✓ Wrote header: {add_header}")
            
            # Write data rows
            for row_idx in range(num_rows):
                row = [column_data[col_idx][row_idx] for col_idx in range(7)]
                writer.writerow(row)
        
        print(f"✓ Merged CSV created: {output_file} ({num_rows} data rows)")
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == '__main__':
    input_dir = sys.argv[1] if len(sys.argv) > 1 else 'split_columns'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'merged_output.csv'
    
    # Parse header if provided
    header = None
    if '--header' in sys.argv:
        header_idx = sys.argv.index('--header')
        if header_idx + 1 < len(sys.argv):
            header = sys.argv[header_idx + 1]
    
    print(f"Merging files from: {input_dir}")
    print(f"Output file: {output_file}")
    
    if header:
        print(f"Header: {header}")
    
    success = merge_csv(input_dir, output_file, header)
    
    if not success:
        sys.exit(1)
