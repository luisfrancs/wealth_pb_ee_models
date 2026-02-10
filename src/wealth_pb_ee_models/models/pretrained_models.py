## VERSION 10 February 2026
## AUTHOR Luis Sigcha

import tensorflow as tf
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent #build relative paths

def multi_taks_7classes_V2():
    # Load model
    loaded_model=tf.keras.models.load_model(_THIS_DIR /'MODEL_nuWIDECNN_AP_AG_2025-11-26_19-19-37_f1score_0.76_betterEE_updated.h5')
    encoding_dict_PB = {0:'Sitting', 1:'Standing', 2:'Walking', 3:'Running', 4:'Sports', 5:'Cycling', 6:'Lying'}
    encoding_dict_EE = {0:'Sedentary', 1:'LPA', 2:'MVPA'}
    return loaded_model, encoding_dict_PB, encoding_dict_EE

def multi_taks_7classes_AP():
    # Load model
    loaded_model = tf.keras.models.load_model(_THIS_DIR /'MODEL_nuWIDECNN_AP_SINGLE_2025-11-21_15-29-12_f1score_0.728_updated.h5')
    encoding_dict_PB = {0:'Sitting', 1:'Standing', 2:'Walking', 3:'Running', 4:'Sports', 5:'Cycling', 6:'Lying'}
    encoding_dict_EE = {0:'Sedentary', 1:'LPA', 2:'MVPA'}
    return loaded_model, encoding_dict_PB, encoding_dict_EE

def multi_taks_7classes_AG():
    # Load model
    loaded_model = tf.keras.models.load_model(_THIS_DIR /'MODEL_nuWIDECNN_AG_SINGLE_2025-12-03_14-26-02_f1score_0.596_model_updated.h5')
    encoding_dict_PB = {0:'Sitting', 1:'Standing', 2:'Walking', 3:'Running', 4:'Sports', 5:'Cycling', 6:'Lying'}
    encoding_dict_EE = {0:'Sedentary', 1:'LPA', 2:'MVPA'}
    return loaded_model, encoding_dict_PB, encoding_dict_EE
