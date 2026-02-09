#

from pygt3x.reader import FileReader
import pandas as pd

def read_gt3x_file(file_path):
    """
    Read GT3X file and convert to DataFrame.
    Args:
        file_path (str): Path to the GT3X file
    Returns:
        pandas.DataFrame: DataFrame containing the accelerometer data
    Raises:
        FileNotFoundError: If the file does not exist
        Exception: If there's an error reading the file
    """
    try:
        # Get accelerometer data

        # load GT3X file
        #logger.info("read_gt3x_file: start reading gt3x")

        with FileReader(file_path) as reader:
            # Get accelerometer data
            df = reader.to_pandas()
            df['time'] = df.index
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.reset_index()
            df = df[['time', 'X', 'Y', 'Z']]
            #logger.info("read_gt3x_file: df finished")

        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"GT3X file not found at path: {file_path}")

    except Exception as e:
        raise Exception(f"Error reading GT3X file: {str(e)}")
        

def read_gt3x_file_with_fs(file_path):
    """
    Read GT3X file and return accelerometer data and sampling rate.
    Args:
        file_path (str): Path to the GT3X file
    Returns:
        df (pd.DataFrame): DataFrame with columns [time, X, Y, Z]
        fs (int): Sampling rate (Hz)
    Raises:
        FileNotFoundError: If file does not exist
        Exception: If reading fails
    """
    try:
        with FileReader(file_path) as reader:
            # Read accelerometer data
            df = reader.to_pandas()
            # Read metadata
            info = reader.info
            # Extract sampling rate (version-independent)
            if hasattr(info, "sample_rate"):
                fs = info.sample_rate
            elif isinstance(info, dict) and "Sample_Rate" in info:
                fs = info["Sample_Rate"]
            else:
                raise ValueError("Sample rate not found in GT3X metadata.")
            # Build time column
            df["time"] = df.index
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.reset_index(drop=True)
            df = df[["time", "X", "Y", "Z"]]
        return df, fs

    except FileNotFoundError:
        raise FileNotFoundError(f"GT3X file not found at path: {file_path}")

    except Exception as e:
        raise Exception(f"Error reading GT3X file: {str(e)}")
