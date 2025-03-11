import pandas as pd

def get_minmax(csv_path):
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Dictionary to store min/max for numeric columns
    minmax = {}
    
    # Check each column
    for col in df.columns:
        # Check if column contains numeric data
        if pd.api.types.is_numeric_dtype(df[col]):
            minmax[col] = {
                'min': df[col].min(),
                'max': df[col].max()
            }
    
    return minmax

def print_minmax(csv_path):
    results = get_minmax(csv_path)
    
    print(f"\nMin/Max values for {csv_path}:")
    for col, values in results.items():
        print(f"{col}:")
        print(f"  Min: {values['min']}")
        print(f"  Max: {values['max']}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        print_minmax(csv_path)
    else:
        print("Please provide a CSV file path as an argument")
