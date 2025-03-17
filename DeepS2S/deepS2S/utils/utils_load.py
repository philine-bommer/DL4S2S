from pathlib import Path

import xarray as xr
import numpy as np
from typing import Tuple


from torch import utils
import lightning.pytorch as pl


# import utils 
from ..dataset.datasets_regimes import TransferData, WeatherDataset
from .utils_build import load_multi_model
from .utils_data import generate_clim_pred, load_data
from .utils_model import test_model_and_data, best_model_folder
from .utils_evaluation import evaluate_accuracy 


def collect_statistics_from_model(pths: str, 
                       exp_dir: str, 
                       config: dict, 
                       mod_name: str, 
                       params: dict, 
                       trainer: pl.Trainer, 
                       test_loader: utils.data.DataLoader, 
                       data_info: dict, 
                       var_comb: list, 
                       seasons: list, 
                       results_directory: Path,
                       targets: np.ndarray, 
                       **kwargs
                       )-> Tuple[dict, np.ndarray, np.ndarray]:

    results = {}
    probabilities = []
    pred_classes = []
    collective_acc = []
    collective_acc_ts = []
    for num in range(1,len(pths)):
        current_dir =  exp_dir + f'run_{num}/'
        
        model, result, _, _ = load_multi_model(config, current_dir, mod_name, **params)

        predictions = result['predictions'].reshape(targets.shape[0],targets.shape[1],4)
        probabilities.append(predictions)

        p_class = np.argmax(predictions, axis = 2)
        pred_classes.append(p_class)
        collective_acc.append(result['mean_acc'])

        results[f"run {num}"]={'predictions':predictions,
                            'class pred':p_class,
                            'result':result['acc'],
                            'acc':result['mean_acc']}
        
        _, fine_acc_ts = evaluate_accuracy(model, trainer, test_loader,config, data_info, var_comb, seasons, 'test')
        collective_acc_ts.append(fine_acc_ts)


    loop_probabilities = np.concatenate(probabilities).reshape(
                        len(probabilities),predictions.shape[0],predictions.shape[1]
                        ,predictions.shape[2])  
    loop_classes = np.concatenate(pred_classes).reshape(
                        len(probabilities),predictions.shape[0],predictions.shape[1]) 
    acc_ts = np.concatenate(collective_acc_ts).reshape(len(pths)-1,6)

    np.savez(str(results_directory/Path(f'accuracy_{len(pths)-1}model.npz')),
            mean_acc= acc_ts.mean(axis = 0),
            var_acc= acc_ts.var(axis = 0), std_acc = acc_ts.std(axis = 0))

    return results, loop_probabilities, loop_classes

def collect_statistics_from_file(pths: str, 
                       exp_dir: str, 
                       results_directory: Path,
                       **kwargs
                       )-> Tuple[dict, np.ndarray, np.ndarray]:

    results = {}
    probabilities = []
    pred_classes = []
    collective_scaled_acc = []
    collective_acc_ts = []
    for num in range(1,len(pths)):
        current_dir =  exp_dir + f'run_{num}/'
        
        calibration_result = np.load(f'{current_dir}calibration_prediction.npz')

        results[f"run {num}"]=calibration_result
        
        predictions = calibration_result['predictions']
        fine_acc_ts = calibration_result['acc_ts']
        scaled_acc_ts = calibration_result['scaled_acc_ts']
        pred_classes.append(calibration_result['scaled_classes'])
        probabilities.append(calibration_result['scaled_predictions'])

        collective_acc_ts.append(fine_acc_ts)
        collective_scaled_acc.append(scaled_acc_ts)


    loop_probabilities = np.concatenate(probabilities).reshape(
                        len(probabilities),predictions.shape[0],predictions.shape[1]
                        ,predictions.shape[2])  
    loop_classes = np.concatenate(pred_classes).reshape(
                        len(probabilities),predictions.shape[0],predictions.shape[1]) 
    acc_ts = np.concatenate(collective_acc_ts).reshape(len(pths)-1,6)

    np.savez(str(results_directory/Path(f'accuracy_{len(pths)-1}model_temp_scale.npz')),
            mean_acc= acc_ts.mean(axis = 0),
            var_acc= acc_ts.var(axis = 0), std_acc = acc_ts.std(axis = 0))

    return results, loop_probabilities, loop_classes