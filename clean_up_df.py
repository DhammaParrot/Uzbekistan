import pandas as pd

import numpy as np
from model_training import find_prefix_years
from helper import add_column_next_tov2

def clean_and_impute_dataframe(df, columns_to_exclude):
  
    for col in df.columns:
        # Skip the conversion and imputation for columns in the exclude list
        if col in columns_to_exclude:
            continue
        
        # Check if the column's type is object (string), then replace '#REF!' with NaN
        # and remove commas before converting to numeric.
        if df[col].dtype == 'object':
            df[col] = df[col].replace('#REF!', np.nan)
            
            df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
    df.replace(0, np.nan, inplace=True)
    for col in df.columns:
        if col not in columns_to_exclude and df[col].isnull().any():
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)
    df_years = find_prefix_years(df, ["Beds","BOR","Admissions"])
   
    for year in range(df_years[0], df_years[-1]+1):
       
        beds_col = f'Beds {year}'
        occupied_col = f'Beds_Occupied {year}'
        add_column_next_tov2(df, occupied_col)
        df[occupied_col] = (df[beds_col] * df[f'BOR {year}'] * 0.01).round().astype(int)
    
    return df




def transform_df_multi_prefix(df, prefixes):
    # Initialize an empty DataFrame to hold all transformed data
    combined_transformed_df = pd.DataFrame()

    # Get the list of all years present in the column names for each prefix
    all_years = []
    common_years_set = None
    for prefix in prefixes:
        prefix_years = df.filter(like=prefix).columns.str.extract(r'\b(\d{4})\b').dropna().astype(int).squeeze().tolist()
        prefix_years_set = set(prefix_years)
      
        all_years.extend(prefix_years)
    all_years = sorted(set(all_years))
    
    # Check the range of years
    if not all_years:
        raise ValueError("No year columns found matching the given prefixes.")
    years = range(min(all_years), max(all_years)-1)
    
    # Transform the DataFrame
    for prefix in prefixes:
        transformed_rows = []
        for _, row in df.iterrows():
            for t in years:
                # Create a dictionary for the new row
                
                new_row = {
                    'Region': row['Region'],
                    'Name': row['Name'],
                    f'{prefix} t-1': row.get(f'{prefix} {t}') if t >= years.start else None,
                    f'{prefix} t': row.get(f'{prefix} {t+1}'),
                    'Year t': t+1  # Adding 'Year t' to be used as a merging key
                }
                # if prefix == "Beds" and t < years.stop:
                if t < years.stop:
                   
                    new_row[f'{prefix} t+1'] = row.get(f'{prefix} {t+2}')
                # Append the new row to the list
                transformed_rows.append(new_row)

        prefix_df = pd.DataFrame(transformed_rows)
        if combined_transformed_df.empty:
            combined_transformed_df = prefix_df
        else:
            combined_transformed_df = pd.merge(combined_transformed_df, prefix_df, on=['Region', 'Name', 'Year t'], how='outer')

    return combined_transformed_df

