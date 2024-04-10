import streamlit as st
import pandas as pd
from clean_up_df import clean_and_impute_dataframe, transform_df_multi_prefix
from model_training import predict_future_years
from helper import load_csv, add_column_next_tov2, filter_columns_by_prefix, columns_filled, make_heatmap
import matplotlib.pyplot as plt
import numpy as np
import re
import altair as alt
# Page configuration
st.set_page_config(
    page_title="Uzbekistan Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")


alt.themes.enable("dark")
#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)


# Path to your pre-uploaded CSV file within the app's directory
#pre_uploaded_file_path = '/Users/tommaso/m4h-kaz/Tommaso_Data2 - 3-R-Overview_v2.csv'

pre_uploaded_file_path = 'Tommaso_Data2 - 3-R-Overview_v2.csv'
# Main function to display the app
def main():
    
    st.image("m4h-logo-web.png")
    st.title("Hospitals Uzbekistan")

    # Create vertical tabs in the sidebar
    
    edited_df = None

    # Initialize session state for the DataFrame
    if 'df' not in st.session_state:
        st.session_state.df = load_csv(pre_uploaded_file_path)
    if 'data_added' not in st.session_state:
        st.session_state.data_added = False 

    # Dashboard Main Panel
    col = st.columns((3.0, 3.5), gap='medium')

    
    with col[0]:
        st.markdown('#### Hospital Beds Occupancy Modelling')
        
        # Add new columns
        st.markdown("#### Add New Column:")
        st.markdown(
            "If you have new data for the Beds and Admissions for a particular year, "
            "and want to predict Bed Occupancy for the following year, add columns like: "
            "Beds <year>, BOR <year>, Admissions <year> and paste in the data for each "
            "column in the editable dataframe below. Then click the Train Model below button to train the model."
        )
        col_name = st.text_input("Column Name:")
        if st.button("Add Column") and 'df' in st.session_state:
            if col_name:
                if col_name not in st.session_state.df.columns:
                    add_column_next_tov2(st.session_state.df, col_name)
                    st.session_state.df[col_name] = np.nan  # Initialize with NaN
                    st.success(f"Column '{col_name}' added.")
                else:
                    st.error(f"Column '{col_name}' already exists.")
        filtered=filter_columns_by_prefix(st.session_state.df, ["Region","Name","Beds","BOR","Admissions"])
        edited_df =st.data_editor(data=filtered,key="first")
        if st.button("Train Model"):
            for col in edited_df.columns:
                st.session_state.df[col] = edited_df[col]
            if 'trained_model' in st.session_state:
                del st.session_state['trained_model']
            if 'accuracy' in st.session_state:
                del st.session_state['accuracy']
            if 'df_with_predictions' in st.session_state:
                del st.session_state['df_with_predictions']
            # Set the data_added flag to True

            st.experimental_rerun()  # Force a rerun to check for all_columns_present
       
     
        column_names=list(st.session_state.df.columns)
       
        # Extract years from column names and check required columns
        year_pattern = re.compile(r'\b(\d{4})\b')
        years = set(year_pattern.findall(" ".join(column_names)))
        # Extract years from column names and check required columns
        
        
        required_columns_sets = [{f'Beds {year}', f'BOR {year}', f'Admissions {year}'} for year in years]

        all_columns_present = all(
            set.issubset(st.session_state.df.columns) and columns_filled(st.session_state.df, set)
            for set in required_columns_sets
        )

        if all_columns_present:
            
            
            # All required columns are present and filled with actual values; you can proceed
            columns_to_exclude = ['Region', 'Name', 'Category']
            
            st.session_state.df = clean_and_impute_dataframe(st.session_state.df, columns_to_exclude)
            filtered=filter_columns_by_prefix(st.session_state.df, ["Region","Name","Beds","BOR","Admissions"])
            #st.data_editor(data=filtered,key="second")
        else:
            # Not all required columns are present and filled for each detected year; instruct the user to add them
            missing_or_unfilled_columns = [
                col for set in required_columns_sets for col in set
                if col not in st.session_state.df.columns or st.session_state.df[col].isna().all() or (st.session_state.df[col] == 0).all()
            ]
            st.warning(
                "Please ensure all required columns for each year are added and filled with actual data before proceeding: "
                + ", ".join(missing_or_unfilled_columns)
            )

        # Display the updated DataFrame
        

        transformed_df = None
        # Apply the function to the DataFrame with the list of prefixes and years
        selected_prefixes = ["Beds_Occupied", "Admissions","Inpatient days","Livebirths","Deliveries"]
        transformed_df = transform_df_multi_prefix(st.session_state.df, selected_prefixes)
        st.session_state['transformed_df'] = transformed_df

        if 'transformed_df' in st.session_state:

            if 'trained_model' not in st.session_state and 'accuracy' not in st.session_state:
                # Train the model
                df_with_predictions, accuracy, trained_model = predict_future_years(st.session_state.df, st.session_state['transformed_df'])
                st.write(f"Trained Model with: {accuracy:.2%} accuracy")
                st.session_state['df_with_predictions'] = df_with_predictions
                st.session_state['trained_model'] = trained_model
                st.session_state['accuracy'] = accuracy
            else:
                trained_model = st.session_state['trained_model']
                accuracy=st.session_state['accuracy']
                st.write(f"Trained Model achieved: {accuracy:.2%} accuracy")

        st.write(f"Edited Data:")
        st.dataframe(st.session_state['df_with_predictions'])
        if 'df_with_predictions' in st.session_state:
    # Select a specific hospital to plot
            hospital_list = st.session_state['df_with_predictions']['Name'].unique()
            selected_hospital = st.selectbox("Select a Hospital:", hospital_list, key='select_hospital')

            if selected_hospital:
                # Fetch the data for the selected hospital
                hospital_data = st.session_state['df_with_predictions'][st.session_state['df_with_predictions']['Name'] == selected_hospital]

                # Extract years and model prediction columns dynamically
                year_columns = [col for col in hospital_data.columns if 'Beds_Occupied' in col and 'Model' not in col]
                model_columns = [col for col in hospital_data.columns if 'Beds_Occupied' in col and 'Model' in col and 'Model_Std' not in col]

                year_columns.sort(key=lambda x: int(x.split(' ')[-1]) if x.split(' ')[-1].isdigit() else float('inf'))
                model_columns.sort(key=lambda x: int(x.split(' ')[-1]) if x.split(' ')[-1].isdigit() else float('inf'))

                # Prepare the data for Altair
                years = [int(col.split(' ')[-1]) for col in year_columns]
                model_years = [int(col.split(' ')[-2]) for col in model_columns]
                beds_occupied = [hospital_data[col].iloc[0] for col in year_columns]
                predictions = [hospital_data[col].iloc[0] for col in model_columns]

                # Create a DataFrame for the Altair chart
                chart_data = pd.DataFrame({
                    'Year': years + [model_years[-1]],
                    'Beds Occupied': beds_occupied + [predictions[-1]],
                    'Type': ['Ground Truth'] * len(years) + ['Model Prediction']
                })

                # Calculate the error bands
                error_percentage = 0.075
                prediction = predictions[-1]
                upper_bound = prediction * (1 + error_percentage)
                lower_bound = prediction * (1 - error_percentage)

                # Define the colors
                purple_rgb = [129, 35, 88]  # Original purple
                bright_purple_rgb = [229, 135, 188]  # Brighter purple

                # Convert RGB to hexadecimal
                def rgb_to_hex(rgb):
                    return "#{:02x}{:02x}{:02x}".format(*rgb)

                purple_hex = rgb_to_hex(purple_rgb)
                bright_purple_hex = rgb_to_hex(bright_purple_rgb)
                line_chart = alt.Chart(chart_data).mark_line(point=True, color=purple_hex).encode(
                    x=alt.X('Year:O', axis=alt.Axis(title='Year')),
                    y=alt.Y('Beds Occupied:Q', axis=alt.Axis(title='Beds Occupied'))
                ).transform_filter(
                    alt.datum.Type == 'Ground Truth'  # Apply the filter for Ground Truth data
                )

                # Create the Altair chart for the prediction line
                prediction_line = alt.Chart(chart_data).mark_line(point=True, color=bright_purple_hex).encode(
                    x='Year:O',
                    y='Beds Occupied:Q'
                ).transform_filter(
                    alt.datum.Type == 'Model Prediction'  # Apply the filter for Model Prediction data
                )

                # Dotted line connecting the last ground truth value with the prediction (using bright purple)
                dotted_line = alt.Chart(pd.DataFrame({
                    'Year': [years[-1], model_years[-1]],
                    'Beds Occupied': [beds_occupied[-1], predictions[-1]]
                })).mark_line(point=False, strokeDash=[5, 5], color=bright_purple_hex).encode(
                    x='Year:O',
                    y='Beds Occupied:Q'
                )

                # Error bands around the prediction (remain unchanged)
                error_bands = alt.Chart(pd.DataFrame({
                    'Year': [model_years[-1]] * 2,
                    'Beds Occupied': [lower_bound, upper_bound]
                })).mark_area(opacity=0.3, color='orange').encode(
                    x='Year:O',
                    y='Beds Occupied:Q'
                )
                # Combine the charts
                combined_chart = alt.layer(line_chart, prediction_line,dotted_line, error_bands).properties(
                    title=f'Beds Occupied - {selected_hospital}',
                    width=750,
                    height=500
                )

                st.altair_chart(combined_chart)
                
              

       

    with col[1]:
        st.markdown('#### Charts and Statistics') 
        # Perform the melt operation
        # Perform the melt operation
        df_melted = pd.melt(st.session_state['df_with_predictions'], id_vars=['Name'], var_name='Year_Variable', value_name='Value')

        # Extract the year and variable name
        df_melted['Year'] = df_melted['Year_Variable'].str.extract('(\d{4})')
        df_melted['Variable'] = df_melted['Year_Variable'].str.extract('([A-Za-z]+)')

        # Drop the original Year_Variable column as it's no longer needed
        df_melted.drop('Year_Variable', axis=1, inplace=True)

        # Now you might want to pivot the table to have one row per year per category
        df_pivoted = df_melted.pivot_table(index=['Name', 'Year'], columns='Variable', values='Value').reset_index()
        
        
        # options = [col for col in df_pivoted.columns if col not in ['Name', 'Year','Population']]
        

        # selected_variable = st.selectbox(
        #     'Select a variable for the heatmap:',
        #     options=options
        # )
       

        # heatmap = make_heatmap(df_pivoted, 'Year', 'Name', selected_variable, 'blues', True)
        # st.altair_chart(heatmap, use_container_width=True)
        options = [col for col in df_pivoted.columns if col not in ['Name', 'Year', 'Population']]

        # Create a list of heatmaps, one for each variable
        heatmaps = [make_heatmap(df_pivoted, 'Year', 'Name', variable, 'viridis', True, title=variable) for variable in options]


        combined_heatmap = alt.vconcat(*heatmaps).resolve_scale(color='independent').configure_axis(
    labelFontSize=12,
    titleFontSize=12
)

        st.altair_chart(combined_heatmap, use_container_width=True)

# Run the main function when the script is executed
if __name__ == '__main__':
    main()
