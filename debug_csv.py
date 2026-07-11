python << 'EOF'
import pandas as pd

df = pd.read_csv(r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer\data\processed\resumes_unified.csv")
print("Columns:", df.columns.tolist())
print()
print("First row sample (first 200 chars per column):")
for col in df.columns[:5]:  # First 5 columns
    val = str(df[col].iloc[0])[:200]
    print(f"\n{col}:")
    print(f"  {val}")
EOF