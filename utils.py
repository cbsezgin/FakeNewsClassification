import pandas as pd


def read_dataframe(path, **kwargs):
    data = pd.read_csv(path, **kwargs)
    return data


def text_stat(data,col):
    data = data[col].str.split().str.len()
    return data.describe()


def remove_columns(df,cols):
    for col in cols:
        if col in df.columns:
            df = df.drop(col,axis=1)
    return df


def null_processing(df):
    cols = (df.columns[df.isnull().sum() > 0])
    df.dropna(inplace=True, axis=1)
    return df


def clean_data(df,cols):
    df = remove_columns(df, cols)
    df = null_processing(df)
    return df
