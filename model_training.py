from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def percentage_mae(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error per target variable
    y_true and y_pred should be of the same 2D shape
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true), axis=0)
def predict_future_years(df, transformed_df):
    # List of predictor variables
    #predictor_variables = ['Beds_Occupied t-1', 'Beds_Occupied t', 'Livebirths t-1', 'Livebirths t', 'Inpatient days t-1', 'Inpatient days t',
     #                      'Deliveries t-1', 'Deliveries t']
    predictor_variables = ['Beds_Occupied t-1', 'Beds_Occupied t', 'Admissions t-1', 'Admissions t']

    # List of target variables
    target_variables = ['Beds_Occupied t+1']

    # Create a copy of the df DataFrame to store predictions
    df_with_predictions = df.copy()

    # Train a separate model for each target variable and add predictions
    prefixes = [var.split()[0] for var in target_variables]

    # Use the find_prefix_years function to find the range of years for each prefix
    df_years = find_prefix_years(df, prefixes)

    for target in target_variables:
        # Define the features (X) and the target (Y)
        X = transformed_df[predictor_variables]
        Y = transformed_df[target]

        # Split the data into training and test sets
        X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.3, shuffle=False)

        # Initialize the model
        model = RandomForestRegressor()

        # Train the model
        model.fit(X_tr.values, Y_tr.values)
        pred=model.predict(X_ts.values)
        accuracy=1 - percentage_mae(pred, Y_ts)
        
        # Iterate over the years in df_years to make predictions
        for i in range(len(df_years) - 2):
            year_t_minus_1 = df_years[i]
            year_t = df_years[i + 1]
            year_t_plus_1 = df_years[i + 2]

            new_data = {
                'Beds_Occupied t-1': df[f'Beds_Occupied {year_t_minus_1}'],
                'Beds_Occupied t': df[f'Beds_Occupied {year_t}'],
                'Admissions t-1': df[f'Admissions {year_t_minus_1}'],
                'Admissions t': df[f'Admissions {year_t}'],
            
            }

            new_df = pd.DataFrame(new_data)
            predictions = model.predict(new_df)
            pred_std = np.std([tree.predict(new_df) for tree in model.estimators_], axis=0)
            # Add predictions to df_with_predictions
            add_column_next_to(df_with_predictions, f"{target} {year_t_plus_1+1} Model", predictions)
            add_column_next_to(df_with_predictions, f"{target} {year_t_plus_1+1} Model_Std", pred_std)

    # Display the DataFrame with added predictions
    return df_with_predictions,accuracy, model


def find_prefix_years(df, prefixes):
    all_years = []
    common_years_set = None
    for prefix in prefixes:
        prefix_years = df.filter(like=prefix).columns.str.extract(r'\b(\d{4})\b').dropna().astype(int).squeeze().tolist()
        prefix_years_set = set(prefix_years)
        
        # Initialize the common_years_set with years from the first prefix
        if common_years_set is None:
            common_years_set = prefix_years_set
        else:
            # Check if the years match for subsequent prefixes
            if common_years_set != prefix_years_set:
                raise ValueError(f"Years for prefix '{prefix}' do not match the common set of years.")
        all_years.extend(prefix_years)
    all_years = sorted(set(all_years))
    return all_years

# Function to add a new column next to similar columns
def add_column_next_to(df, new_col_name, new_col_data):
    prefix = new_col_name.split(' ')[0]
    
    similar_cols = [col for col in df.columns if col.startswith(prefix)]
    if similar_cols:
        last_col_position = df.columns.get_loc(similar_cols[-1])
       
        df.insert(last_col_position + 1, new_col_name, new_col_data)
    else:
        df[new_col_name] = new_col_data