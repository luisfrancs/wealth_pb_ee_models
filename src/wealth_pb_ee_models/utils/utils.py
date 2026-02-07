## VERSION 28 NOV 2025
## AUTHOR Luis Sigcha

## Load the required packages
#import mat73
import os
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from wealth_pb_ee_models.utils import uos_update_LS

## Load data utils
def load_data_AP_AG_CSV(file_name="file.cvs"):
    data = np.loadtxt(file_name, delimiter=",", skiprows=2,usecols=(0,1,2,3,4,5,6),max_rows=None)#modify to accpet all data
    time_stamps=data[:,0]
    X=data[:,1:7]
    return X,time_stamps

def load_data_AP_CSV(file_name="file.csv"):
    '''Load an ActivPAL file from CSV format (exported as uncompresed csv).
    Returns: X triaxial sensor data; time_stamps'''
    data = np.loadtxt(file_name, delimiter=";", skiprows=2,usecols=(0,2,4,3),max_rows=144000)#modify to accpet all data
    time_stamps=data[:,0]
    X=data[:,1:4]
    return X,time_stamps

def load_data_AP_DATX(file_name="file.datx"):
    '''Load an ActivPAL file from a compressed .datx file
        Returns: X triaxial sensor data; time_stamps'''
    # it uses an adapted version of the uos library to open directly the files
    activpal_data = uos_update_LS.ActivpalData(file_name)
    time_stamps=activpal_data.timestamps
    time_stamps=time_stamps.to_numpy()
    X = activpal_data.signals.to_numpy() #get fullscale data as numpy
    #X[:, [2, 1]] = X[:, [1, 2]]#swap axis to harmonize with PALAnalysis export
    return X,time_stamps

#Predict PB and EE utils

def predict_PA(X, loaded_model, scaling_factor=1,window_size=200,step_size=200,batch_size=512):
    '''Makes predictions usign a triaxial data
    Returns: Prediciton as probabilty'''
    X=X/scaling_factor #normalize data
    X_split = sliding_window_triaxial(X, window_size=window_size, step_size=step_size)#Slinding window
    y_pred = loaded_model.predict(X_split,batch_size=batch_size)#Predic over all data file
    return y_pred

def sliding_window_triaxial(signal, window_size, step_size):
    """step_size (int): Step size between windows.
    Returns: Array of sliding windows of shape (num_windows, window_size, 3).    """
    #Get number of channels (deafult 3)
    channels=signal.shape[-1]
    # Calculate the number of windows
    num_windows = (signal.shape[0] - window_size) // step_size + 1
    if num_windows <= 0:
        raise ValueError("Invalid combination of window size and step size for the given signal length.")
    # Create sliding windows
    windows = np.lib.stride_tricks.sliding_window_view(signal, (window_size, channels))
    # Extract and reshape windows to remove the singleton dimension
    windows = windows[::step_size, :, :]  # Select windows with the specified step size
    windows = windows.reshape(-1, window_size, channels)  # Reshape to (num_windows, window_size, 3)
    return windows

def post_process(y_predict_proba):
    ''' Implements: post-processing tasks to Predicition as probability
    Returns: hard predictions encoded as integers '''
    y_pred=np.argmax(y_predict_proba, axis=1)#get activity class from probability
    ##TO-DO MEDIAN FILTER
    return y_pred

#Single file procesing
def predict_single_file_AP_v1(file_name, model,encoding_dict_PB,encoding_dict_EE,sensor="AP",batch_size=512):
    ''' Performs a complete prediction pipe-line to a single file (ActivPAL 20 Hz from a .CSV uncrompressed file)
    Returns: Window based prediction as pandas dataframe with columns Time (time-stamps) and activity (as string) '''
    window_size=200 #60=3 seconds at 20 Hz
    X,time_stamps=load_data_AP_CSV(file_name=file_name)#load AP data file
    y_predict_proba=predict_PA(X, model,scaling_factor=1,window_size=window_size,overlap=0)#predict
    #Physical Behaviour
    y_predict_proba_PB=y_predict_proba['task_1']
    predictions_PB=post_process(y_predict_proba_PB)#POST-PROCESSING
    #ENERGY EXPENDITURE
    y_predict_proba_EE=y_predict_proba['task_2']
    predictions_EE=post_process(y_predict_proba_EE)#POST-PROCESSING
    # Convert window prediction to df (window based prediction)
    croped_shape=window_size*predictions_PB.shape[0]
    cp_time_stamps = time_stamps[0:croped_shape]#crop to size
    nu_time_stamps=cp_time_stamps[::window_size]#downsample time stamps
    X=X[0:croped_shape,:]#crop to size
    # Convert the NumPy array to a Pandas DataFrame
    df = pd.DataFrame({'Time': nu_time_stamps, 'activity': predictions_PB, 'Energy_expenditure': predictions_EE})
    #Transform using the mapping dictionary
    df['activity'] = df['activity'].map(encoding_dict_PB)#encode numeric predictions to class
    df['Energy_expenditure'] = df['Energy_expenditure'].map(encoding_dict_EE)#encode numeric predictions to class
    return df, X
### ActivPAL
def predict_single_file_AP(file_name, model, encoding_dict_PB,  encoding_dict_EE, batch_size=512):
    """
    Performs a complete prediction pipeline for a single activPAL file.
    Supported extensions: .csv, .datx

    Returns:
        df (pd.DataFrame): Window-based predictions with columns:
            - Time
            - activity (mapped to class names via encoding_dict_PB)
            - Energy_expenditure (mapped to class names via encoding_dict_EE)
        X (np.ndarray): Cropped signal aligned to the returned windows

    Error handling:
        - If path is not a file: raises FileNotFoundError
        - If extension is not supported: raises ValueError
        - If processing fails: raises RuntimeError
    """
    try:
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError("Processing error: 'file_name' must be a non-empty string.")

        if not os.path.isfile(file_name):
            raise FileNotFoundError( f"Processing error: '{file_name}' is not a file or does not exist." )

        _, ext = os.path.splitext(file_name)
        ext = ext.lower()
        window_size = 200  # 20 Hz → 10 s windows
        overlap=0
        # Load according to extension
        if ext == ".csv":
            X, time_stamps = load_data_AP_CSV(file_name=file_name)
            y_predict_proba = predict_PA(X, model, scaling_factor=1, window_size=window_size, overlap=overlap)

        elif ext == ".datx":
            warnings.simplefilter("ignore")
            X, time_stamps = load_data_AP_DATX(file_name=file_name)
            y_predict_proba = predict_PA(X, model, scaling_factor=1, window_size=window_size, step_size=window_size)

        else:
            raise ValueError(
                f"Processing error: unsupported file extension '{ext}'. "
                "Only '.csv' and '.datx' are supported." )

        # Physical Behaviour
        y_predict_proba_PB = y_predict_proba["task_1"]
        predictions_PB = post_process(y_predict_proba_PB)
        # Energy Expenditure
        y_predict_proba_EE = y_predict_proba["task_2"]
        predictions_EE = post_process(y_predict_proba_EE)
        # Window alignment
        croped_shape = window_size * predictions_PB.shape[0]
        cp_time_stamps = time_stamps[:croped_shape]
        nu_time_stamps = cp_time_stamps[::window_size]
        X = X[:croped_shape, :]
        # Build dataframe
        df = pd.DataFrame({
            "Time": nu_time_stamps,
            "activity": predictions_PB,
            "Energy_expenditure": predictions_EE
        })
        # Map to class labels
        df["activity"] = df["activity"].map(encoding_dict_PB)
        df["Energy_expenditure"] = df["Energy_expenditure"].map(encoding_dict_EE)
        return df, X

    except (FileNotFoundError, ValueError, KeyError):
        raise
    except Exception as e:
        raise RuntimeError(f"Processing error: {e}") from e
### actiGraph

### Dual sesnsor (ActivPAL+ actiGraph)
def predict_file_AP_AG_CSV(file_name, model,encoding_dict_PB,encoding_dict_EE,batch_size=512):
    ''' Performs a complete prediction pipe-line to a single file (ActivPAL and ActiGrap sincronized at 20 Hz)
    Returns: Window based prediction as pandas dataframe with columns Time (tiem-stamps) and activity (as string) '''
    warnings.simplefilter('ignore')
    window_size=200 #60=3 seconds at 20 Hz
    X,time_stamps=load_data_AP_AG_CSV(file_name=file_name)#load AP data file
    y_predict_proba=predict_PA(X, model,scaling_factor=1,window_size=window_size,step_size=window_size)#predict
    #Physical Behaviour
    y_predict_proba_PB=y_predict_proba['task_1']
    predictions_PB=post_process(y_predict_proba_PB)#POST-PROCESSING
    #ENERGY EXPENDITURE
    y_predict_proba_EE=y_predict_proba['task_2']
    predictions_EE=post_process(y_predict_proba_EE)#POST-PROCESSING
    # Convert window prediction to df (window based prediction)
    croped_shape=window_size*predictions_PB.shape[0]
    cp_time_stamps = time_stamps[0:croped_shape]#crop to size
    nu_time_stamps=cp_time_stamps[::window_size]#downsample time stamps
    X=X[0:croped_shape,:]#crop to size
    # Convert the NumPy array to a Pandas DataFrame
    df = pd.DataFrame({'Time': nu_time_stamps, 'activity': predictions_PB, 'Energy_expenditure': predictions_EE})
    #Transform using the mapping dictionary
    df['activity'] = df['activity'].map(encoding_dict_PB)#encode numeric predictions to class
    df['Energy_expenditure'] = df['Energy_expenditure'].map(encoding_dict_EE)#encode numeric predictions to class
    return df, X
