import numpy as np
import pandas as pd
from pymannkendall import original_test as mk_test

# Load the CSV you already exported
df = pd.read_csv('./outputs/csv/per_crop_mean_suitability.csv', index_col='Year')

series = df['Overall'].values
mk = mk_test(series)

print(f"Sen's slope:  {mk.slope:.6f} class units / year")
print(f"Intercept:    {mk.intercept:.6f} (value at index 0 = year 1979)")
print(f"p-value:      {mk.p:.4f}")
print(f"Tau:          {mk.Tau:.3f}")
print(f"Significant:  {mk.p < 0.05}")

# Reconstruct the trend line to verify
years = df.index.values
trend_at_1979 = mk.intercept
trend_at_2018 = mk.intercept + mk.slope * (len(series) - 1)
print(f"\nTrend line value at 1979: {trend_at_1979:.4f}")
print(f"Trend line value at 2018: {trend_at_2018:.4f}")
print(f"Total change over 40 years: {trend_at_2018 - trend_at_1979:.4f} class units")