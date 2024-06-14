import pandas as pd

def load_and_preprocess_data(file_path, selected_vars):
    df = pd.read_excel(file_path)
    df_clean = df.dropna(subset=selected_vars)  # Drop rows with NaNs in selected variables
    
    if all(var in df_clean.columns for var in selected_vars):
        df_selected = df_clean[selected_vars].apply(pd.to_numeric, errors='coerce')
        df_selected = df_selected.dropna()  # Drop rows with NaNs after converting to numeric
        return df_selected
    else:
        raise ValueError("Some selected variables are not in the dataframe columns.")

def get_summary_stats(df):
    return df.describe()

def get_correlation_matrix(df):
    return df.corr()
