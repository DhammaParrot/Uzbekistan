# Function to load and return the CSV file as a pandas DataFrame
import pandas as pd
import altair as alt
def load_csv(file_path):
    return pd.read_csv(file_path)

# Function to add a new column next to columns with the same prefix
def add_column_next_tov2(df, new_col_name):
    if new_col_name in df.columns:
        return 
    prefix = new_col_name.split(' ')[0]
    similar_cols = [col for col in df.columns if col.startswith(prefix)]
    if similar_cols:
        last_col_position = df.columns.get_loc(similar_cols[-1])
        df.insert(last_col_position + 1, new_col_name, "")
    else:
        df[new_col_name] = ""

def filter_columns_by_prefix(df, prefixes):
        # Use a list comprehension to filter column names that start with any of the prefixes
            filtered_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes)]
            return df[filtered_cols]
def columns_filled(df, columns):
        for col in columns:
            if df[col].isna().all() or (df[col] == 0).all():  # You can add more placeholder checks here
                return False
        return True



# def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme, show_legend=True):
#     color_encoding = alt.Color(f'max({input_color}):Q',
#                                scale=alt.Scale(scheme=input_color_theme))
    
#     if not show_legend:
#         color_encoding = color_encoding.encode(legend=None)

#     heatmap = alt.Chart(input_df).mark_rect().encode(
#         y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Year", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
#         x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
#         color=color_encoding,
#         stroke=alt.value('black'),
#         strokeWidth=alt.value(0.25),
#     ).properties(width=900
#     ).configure_axis(
#         labelFontSize=12,
#         titleFontSize=12
#     )

#     return heatmap

def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme, show_legend=True, title=None):
    color_encoding = alt.Color(f'max({input_color}):Q',
                               scale=alt.Scale(scheme=input_color_theme))
    
    if not show_legend:
        color_encoding = color_encoding.encode(legend=None)

    heatmap = alt.Chart(input_df).mark_rect(
        color='rgb' +str((129, 35, 88))
    ).encode(
        y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Year", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
        x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
        color=color_encoding,
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.25),
    ).properties(width=900, title=title)

    return heatmap
