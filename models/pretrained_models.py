## VERSION 3 DECEMBER 2025
## AUTHOR Luis Sigcha

from tensorflow.keras.models import model_from_json
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent #build relative paths

def multi_taks_7classes_V2():
    # Load model (V2 vesion has a better EE fscore)
    with open(_THIS_DIR /"MODEL_nuWIDECNN_AP_AG_macmetal_2025-11-26_19-19-37_f1score_0.76_betterEE_model_architecture.json", "r") as json_file:
        loaded_model = model_from_json(json_file.read())
    # Load weigths
    loaded_model.load_weights(_THIS_DIR /'MODEL_nuWIDECNN_AP_AG_macmetal_2025-11-26_19-19-37_f1score_0.76_betterEE.weights.h5')
    encoding_dict_PB = {0:'Sitting', 1:'Standing', 2:'Walking', 3:'Running', 4:'Sports', 5:'Cycling', 6:'Lying'}
    encoding_dict_EE = {0:'Sedentary', 1:'LPA', 2:'MVPA'}
    return loaded_model, encoding_dict_PB, encoding_dict_EE

def multi_taks_7classes_AP():
    # Load model
    with open(
        _THIS_DIR /"MODEL_nuWIDECNN_AP_SINGLE_macmetal_2025-11-21_15-29-12_f1score_0.728_model_architecture.json", "r") as json_file:
        loaded_model = model_from_json(json_file.read())
    # Load weigths
    loaded_model.load_weights(_THIS_DIR /'MODEL_nuWIDECNN_AP_SINGLE_macmetal_2025-11-21_15-29-12_f1score_0.728.weights.h5')
    encoding_dict_PB = {0:'Sitting', 1:'Standing', 2:'Walking', 3:'Running', 4:'Sports', 5:'Cycling', 6:'Lying'}
    encoding_dict_EE = {0:'Sedentary', 1:'LPA', 2:'MVPA'}
    return loaded_model, encoding_dict_PB, encoding_dict_EE

def multi_taks_7classes_AG():
    # Load model
    with open(
        _THIS_DIR /"MODEL_nuWIDECNN_AG_SINGLE_macmetal_2025-12-03_14-26-02_f1score_0.596_model_architecture.json", "r") as json_file:
        loaded_model = model_from_json(json_file.read())
    # Load weigths
    loaded_model.load_weights(_THIS_DIR /'MODEL_nuWIDECNN_AG_SINGLE_macmetal_2025-12-03_14-26-02_f1score_0.596.weights.h5')
    encoding_dict_PB = {0:'Sitting', 1:'Standing', 2:'Walking', 3:'Running', 4:'Sports', 5:'Cycling', 6:'Lying'}
    encoding_dict_EE = {0:'Sedentary', 1:'LPA', 2:'MVPA'}
    return loaded_model, encoding_dict_PB, encoding_dict_EE

