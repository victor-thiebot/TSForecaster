import warnings
import pandas as pd


def import_data_from_csv(file_path, frequency=None):
    try:
        return read_time_series_csv(file_path, frequency)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except AssertionError as e:
        print(f"Assertion Error: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return None


# def read_time_series_csv(file_path, frequency, fill_method="linear"):
#     df = pd.read_csv(file_path)
#     assert "Date" in df.columns, "The CSV file must contain a 'Date' column."
#     df["Date"] = pd.to_datetime(df["Date"])
#     assert df["Date"].notnull().all(), "The 'Date' column must not contain null values."
#     df = df.sort_values("Date").reset_index(drop=True)

#     # Interpolate missing values
#     if fill_method is not None:
#         df = df.set_index("Date")
#         df = df.interpolate(method=fill_method)
#         df = df.reset_index()

#     if frequency is not None:
#         df = df.set_index("Date").resample(frequency).mean().reset_index()
#     return df


def read_time_series_csv(file_path, frequency=None, date_column="Date"):
    # Load the data
    df = pd.read_csv(file_path)

    # Convert the date column to datetime format
    df = convert_date_column(df, date_column)

    # Sort the data by the date column
    df = sort_data(df, date_column)

    # Set the date column as the index
    df = set_index(df, date_column)

    # Handle the frequency
    df, frequency = handle_frequency(df, frequency)

    # Interpolate missing values
    df = interpolate_missing_values(df)

    return df, frequency


def convert_date_column(data, date_column):
    if date_column in data.columns:
        data[date_column] = pd.to_datetime(data[date_column])
    return data


def sort_data(data, date_column):
    data = data.sort_values(by=date_column)
    return data


def set_index(data, date_column):
    data = data.set_index(date_column)
    return data


def infer_frequency(data):
    return pd.infer_freq(data.index)


def handle_frequency(data, frequency):
    if frequency is None:
        frequency = infer_frequency(data)
        if frequency is None:
            warnings.warn(
                "Failed to infer the frequency of the time series. Please check the date column and consider providing the frequency explicitly."
            )
        else:
            data = data.asfreq(frequency)
    return data, frequency


def interpolate_missing_values(data):
    data = data.interpolate()
    return data
