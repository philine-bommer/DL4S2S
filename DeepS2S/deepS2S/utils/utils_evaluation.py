# pass in: model, loader, criterion, test data loader, metrics_to_compute
# run model across entire test data and compute given metrics
# use utils model to generate plots and store them
import torch
import lightning as pl
import pdb

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd

from datetime import datetime
import matplotlib.pyplot as plt

from ..dataset.datasets_regimes import WeatherDataset



def balanced_accuracy(pds, tgs):
    ts = tgs.shape[1]
    acc = []
    for t in range(ts): 
        bacc = balanced_accuracy_score(tgs[:,t].flatten().astype(int), 
                                       pds[:,t].flatten().astype(int))
        acc.append(bacc)
    bas = balanced_accuracy_score(tgs.flatten().astype(int), 
                                  pds.flatten().astype(int))
    return np.asarray(acc), bas

def get_targets(test_set, predictions):

     
    targets = []
    for input, output in test_set:
        targets.append(output.cpu().detach().numpy().squeeze())

    try:
        targets = np.concatenate(targets).reshape((predictions.shape[0],
                                               predictions.shape[1]))
    except:
        targets = np.concatenate(targets).reshape((predictions.shape[0],1))


    return targets

def evaluate_accuracy(model: pl.LightningModule,
                      trainer: pl.Trainer,
                      test_loader: pl.LightningDataModule,
                      config: dict, 
                      data_info: dict, 
                      var_comb: dict, 
                      seasons: dict,
                      set: str):

    pred = []
    for i, batch in enumerate(test_loader): 
        pred.append(model.predict_step(batch, i).detach().numpy())

    predictions = []
    for i in range(len(pred)):
        predictions.append(pred[i])

    predictions = np.concatenate(predictions)
    targets = get_targets(test_loader, predictions)

    try:
        acc_per_ts, acc = balanced_accuracy(np.argmax(predictions, axis=-1), targets)
    except:
        acc_per_ts, acc = balanced_accuracy(np.argmax(predictions, axis=-1)[:,None], targets)

    return acc, acc_per_ts

def numpy_predict(model, test_loader):
    """
    Predicts the output using the given model and test data.
    Parameters:
    - model: The trained model used for prediction.
    - test_loader: The data loader containing the test data.
    Returns:
    - predictions: Numpy array containing the predicted outputs.
    """

    pred = []
    for i, batch in enumerate(test_loader): 
        pred.append(model.predict_step(batch, i).detach().numpy())

    predictions = []
    for i in range(len(pred)):
        predictions.append(pred[i])

    predictions = np.concatenate(predictions)
    return predictions

def brier_score(pds, tgs, timestep=None):
    """ Computes Brier Score
        Best score: 0
        Worst score : 1

    Args:
        predictions (np.ndarray): Shape: (samples, timesteps, classes) or (samples, timesteps)
        targets (np.ndarray): Shape: (samples, timesteps)
        timestep (int): Timestep for which to calculate the brier score
    """
    if pds.ndim == 3:
        n_samples, n_timesteps, n_classes = pds.shape
    elif pds.ndim == 2:
        n_samples, n_timesteps = pds.shape
        n_classes = np.max(tgs) + 1
        pds = np.eye(n_classes)[pds]

    tgs = np.eye(n_classes)[tgs]

    brier_per_timestep = (1/(n_classes*n_samples)) * np.sum(np.square(pds - tgs), axis=(0,2))



    if timestep is not None:
        return brier_per_timestep[timestep]
    else:
        return brier_per_timestep
    
def brier_score_weighted(pds, tgs, timestep=None):
    """ Computes Brier Score
        Best score: 0
        Worst score : 1

    Args:
        predictions (np.ndarray): Shape: (samples, timesteps, classes) or (samples, timesteps)
        targets (np.ndarray): Shape: (samples, timesteps)
        timestep (int): Timestep for which to calculate the brier score
    """
    if pds.ndim == 3:
        n_samples, n_timesteps, n_classes = pds.shape
        counts = []
        pdsc = np.argmax(pds, axis = 2)
        for t in range(n_timesteps):
            nums, cts = np.unique(tgs[:,t], return_counts=True)
            if len(cts) < 4:
                class_arr = np.zeros((n_classes,))
                class_arr[nums] = cts
                counts.append(class_arr)
            else:    
                counts.append(cts)
        
    elif pds.ndim == 2:
        n_samples, n_timesteps = pds.shape
        n_classes = np.max(tgs) + 1
    
        counts = []
        for t in range(n_timesteps):
            nums, cts = np.unique(tgs[:,t], return_counts=True)
            if len(cts) < 4:
                class_arr = np.zeros((n_classes,))
                class_arr[nums] = cts
                counts.append(class_arr)
            else:    
                counts.append(cts)
        
        pds = np.eye(n_classes)[pds]
    
    
    tgs = np.eye(n_classes)[tgs]
    counts = np.asarray(counts)
    trues = tgs == pds
    counts = np.abs((trues.sum(axis = 0) - counts)) + trues.sum(axis = 0)
    
    norm = (1/(counts))

    brier_per_class = norm * np.sum(np.square(pds - tgs), axis=0)
    brier_per_timestep = np.mean(brier_per_class, axis=1)



    if timestep is not None:
        return brier_per_timestep[timestep]
    else:
        return brier_per_timestep


def classwise_accuracy(pds, tgs, rgs):
    """ Calculation of class-wise accuracy
    
        Args:
        pds (np.ndarray): Shape: (samples, timesteps, classes) or (samples, timesteps)
        tgs (np.ndarray): Shape: (samples, timesteps)
    """

    # convert class labels to one-hot-encoded 
    n_classes = int(np.max(tgs) + 1)
    n_ts = tgs.shape[1]
    preds = np.eye(n_classes)[pds.astype(int)]
    tars = np.eye(n_classes)[tgs.astype(int)]

    acc = {}
    week_index = []
    
    for j in range(n_ts):
        week_index.append(f'week {j+1}')
        acc_j = np.zeros((n_classes,1))
        for i in range(n_classes):
            pd_ij = preds[:,j,i]
            tg_ij = tars[:,j,i]
    
            correct_ij = np.sum(np.array(pd_ij*tg_ij))
            acc_j[i,0] = correct_ij/np.sum(tg_ij)
            
        acc[week_index[j]] = acc_j.flatten()

    Acc = pd.DataFrame(acc, index = rgs)
    return Acc

def calculate_precision(pds, tgs, rgs):
    """ Calculation of class-wise accuracy
    
        Args:
        pds (np.ndarray): prediction class labels Shape: (samples, timesteps, classes) or (samples, timesteps)
        tgs (np.ndarray): Shape: (samples, timesteps)
    """
    
    # Create Pandas:
    prcs = {}
    average_prcs = []
    week_index = []
    
    for i in range(6):
        # how often was each class predicted in this time step
        precisions = []
        week_index.append(f'week {i+1}')
        for c in range(4):
            c_indices = [idx for idx, cl in enumerate(tgs[:, i]) if cl == c]
            not_c_indices = [idx for idx, cl in enumerate(tgs[:, i]) if cl != c]
            true_positives = np.sum(pds[c_indices, i] == tgs[c_indices, i])
            false_positives = np.sum(pds[not_c_indices, i] == c)
            if (true_positives+false_positives) == 0:
                precision = 0
            else:
                precision = true_positives / (true_positives+false_positives)
            precisions.append(precision)
        prcs[week_index[i]] = precisions 
        average_prcs.append(np.mean(np.asarray(precisions)))
    
    Prcs = pd.DataFrame(prcs, index = rgs)
    average_prcs = np.asarray(average_prcs)
    
    return Prcs, average_prcs

def brier_score_sklearn(pds, tgs, timestep=None):
    """ Computes Brier Score
        Best score: 0
        Worst score : 1

    Args:
        predictions (np.ndarray): Shape: (samples, timesteps, classes) or (samples, timesteps)
        targets (np.ndarray): Shape: (samples, timesteps)
        timestep (int): Timestep for which to calculate the brier score
    """
    if pds.ndim == 3:
        n_samples, n_timesteps, n_classes = pds.shape
        counts = []
                
        
    elif pds.ndim == 2:
        n_samples, n_timesteps = pds.shape
        n_classes = np.max(tgs) + 1
    
        pds = np.eye(n_classes)[pds]
        
    
    brier_per_timestep = []
    tgs = np.eye(n_classes)[tgs]
    for ts in range(n_timesteps):
        y_true = tgs[:,ts,:].flatten()
        y_pred = pds[:,ts,:].flatten()
        brier_per_timestep.append(brier_score_loss(y_true, y_pred))

    brier_per_timestep = np.array(brier_per_timestep)
    if timestep is not None:
        return brier_per_timestep[timestep]
    else:
        return brier_per_timestep
    

def brier_skill_score(pred1, pred2, tgs, weighted = False, sklrn = True):
    if sklrn:
        if not weighted:
            bs1 = brier_score_sklearn(pred1, tgs)
            bs2 = brier_score_sklearn(pred2, tgs)
        else:
            bs1 = brier_score_sklearn_weighted(pred1, tgs)
            bs2 = brier_score_sklearn_weighted(pred2, tgs)
    else:
        if not weighted:
            bs1 = brier_score(pred1, tgs)
            bs2 = brier_score(pred2, tgs)
        else:
            bs1 = brier_score_weighted(pred1, tgs)
            bs2 = brier_score_weighted(pred2, tgs)
    print(bs1, bs2)
    return 1 - (bs1 / bs2)
    
def calculate_tpr_fpr(y_real, y_pred):
    '''
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations
    
    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes
        
    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    '''
    
    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    
    # Calculates tpr and fpr
    tpr =  TP/(TP + FN) # sensitivity - true positive rate
    fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
    
    return tpr, fpr

def get_all_roc_coordinates(y_real, y_proba):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a threshold for the predicion of the class.
    
    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.
        
    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list

def brier_score(pds, tgs, timestep=None):
    """ Computes Brier Score
        Best score: 0
        Worst score : 1

    Args:
        predictions (np.ndarray): Shape: (samples, timesteps, classes) or (samples, timesteps)
        targets (np.ndarray): Shape: (samples, timesteps)
        timestep (int): Timestep for which to calculate the brier score
    """
    if pds.ndim == 3:
        n_samples, n_timesteps, n_classes = pds.shape
    elif pds.ndim == 2:
        n_samples, n_timesteps = pds.shape
        n_classes = np.max(tgs) + 1
        pds = np.eye(n_classes)[pds]

    tgs = np.eye(n_classes)[tgs]

    # brier_per_timestep = (1/(2*n_samples)) * np.sum(np.square(predictions - targets), axis=(0,2)) 
    brier_per_timestep = (1/(n_classes*n_samples)) * np.sum(np.square(pds - tgs), axis=(0,2))



    if timestep is not None:
        return brier_per_timestep[timestep]
    else:
        return brier_per_timestep

def get_all_roc_coordinates(y_real, y_proba):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a threshold for the predicion of the class.
    
    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.
        
    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return tpr_list, fpr_list

def CSI_multiclass(predictions, targets, timestep=None):
    """ Computes Critical Success Index
        Best score: 0
        Worst score : 1

    Args:
        predictions (np.ndarray): Shape: (samples, timesteps, classes) or (samples, timesteps)
        targets (np.ndarray): Shape: (samples, timesteps)
        timestep (int): Timestep for which to calculate the brier score
    """
    n_classes = int(np.max(targets) + 1)
    preds = np.eye(n_classes)[predictions.astype(int)]

    tars = np.eye(n_classes)[targets.astype(int)]
    csi = np.zeros((targets.shape[1],))
    for i in range(n_classes):
        pred = preds[:,:,i]
        target = tars[:,:,i]
        for j in range(target.shape[1]):
            tp = pred[target[:,j].astype(bool),j].sum()
            fn = target[:,j].sum() - tp
            # rev_targ = np.abs(target[:,j] - 1)
            fp = pred[:,j].sum() - tp #pred[rev_targ.astype(bool),j].sum() 
            if i == 0: 
                csi[j] = (tp/(tp + fn +fp))*(target[:,j].sum()/len(target[:,j]))
            else:
                csi[j] += (tp/(tp + fn +fp))*(target[:,j].sum()/len(target[:,j]))
            # if j == 0 and i >0:
            #     csi[j-1] = csi[j-1]/n_classes 

    if timestep is not None:
        return csi[timestep]
    else:
        return csi
    
def brier_score_sklearn_class(pds, tgs, timestep=None):
    """ Computes Brier Score
        Best score: 0
        Worst score : 1

    Args:
        predictions (np.ndarray): Shape: (samples, timesteps, classes) or (samples, timesteps)
        targets (np.ndarray): Shape: (samples, timesteps)
        timestep (int): Timestep for which to calculate the brier score
    """
    if pds.ndim == 3:
        n_samples, n_timesteps, n_classes = pds.shape
        counts = []
        pdsc = np.argmax(pds, axis = 2)
        for t in range(n_timesteps):
            nums, cts = np.unique(tgs[:,t], return_counts=True)
            if len(cts) < 4:
                class_arr = np.zeros((n_classes,))
                class_arr[nums] = cts
                counts.append(class_arr)
            else:    
                counts.append(cts)
        
    elif pds.ndim == 2:
        n_samples, n_timesteps = pds.shape
        n_classes = np.max(tgs) + 1
    
        counts = []
        for t in range(n_timesteps):
            nums, cts = np.unique(tgs[:,t], return_counts=True)
            if len(cts) < 4:
                class_arr = np.zeros((n_classes,))
                class_arr[nums] = cts
                counts.append(class_arr)
            else:    
                counts.append(cts)
        
        pds = np.eye(n_classes)[pds]
    
    # counts = []
    # for t in range(n_timesteps):
    #     _, cts = np.unique(pds[:,t], return_counts=True)
    #     counts.append(cts)
    
    tgs = np.eye(n_classes)[tgs]
    counts = np.asarray(counts)
    trues = tgs == pds
    counts = np.abs((trues.sum(axis = 0) - counts)) + trues.sum(axis = 0)
    
    norm = (1/(counts))
    
    brier_per_class = []
    for ts in range(n_timesteps):
        for cs in range(n_classes):
            nms = norm[ts,cs]
            y_true = tgs[:,ts,cs].flatten()
            y_pred = pds[:,ts,cs].flatten()
            brier_per_class.append(nms * n_samples * brier_score_loss(y_true, y_pred))
    
    brier_per_class = np.reshape(np.array(brier_per_class),(n_timesteps, n_classes))


    if timestep is not None:
        return brier_per_class[timestep,:]
    else:
        return brier_per_class

def brier_score_class(pds, tgs, timestep=None):
    """ Computes Brier Score
        Best score: 0
        Worst score : 1

    Args:
        predictions (np.ndarray): Shape: (samples, timesteps, classes) or (samples, timesteps)
        targets (np.ndarray): Shape: (samples, timesteps)
        timestep (int): Timestep for which to calculate the brier score
    """
    if pds.ndim == 3:
        n_samples, n_timesteps, n_classes = pds.shape
        counts = []
        pdsc = np.argmax(pds, axis = 2)
        for t in range(n_timesteps):
            nums, cts = np.unique(tgs[:,t], return_counts=True)
    
            if len(cts) < 4:
                class_arr = np.zeros((n_classes,))
                class_arr[nums] = cts
                counts.append(class_arr)
            else:    
                counts.append(cts)
        
    elif pds.ndim == 2:
        n_samples, n_timesteps = pds.shape
        n_classes = np.max(tgs) + 1

        counts = []
        for t in range(n_timesteps):
            nums, cts = np.unique(tgs[:,t], return_counts=True)
            if len(cts) < 4:
                class_arr = np.zeros((n_classes,))
                class_arr[nums] = cts
                counts.append(class_arr)
            else:    
                counts.append(cts)
        
        pds = np.eye(n_classes)[pds]
    
    # counts = []
    # for t in range(n_timesteps):
    #     _, cts = np.unique(pds[:,t], return_counts=True)
    #     counts.append(cts)
    tgs = np.eye(n_classes)[tgs]
    counts = np.asarray(counts)
    trues = tgs == pds
    counts = np.abs((trues.sum(axis = 0) - counts)) + trues.sum(axis = 0)
    
    norm = (1/(counts))
    
    # brier_per_timestep = (1/(2*n_samples)) * np.sum(np.square(predictions - targets), axis=(0,2))
    brier_per_class = norm * np.sum(np.square(pds - tgs), axis=0)
    print(brier_per_class)
    if timestep is not None:
        return brier_per_class[timestep,:]
    else:
        return brier_per_class

def bss_class(pred1, pred2, tgs, sklrn = True):

    if sklrn:
        bs1 = brier_score_sklearn_class(pred1, tgs)
        bs2 = brier_score_sklearn_class(pred2, tgs)
    else:
        bs1 = brier_score_class(pred1, tgs)
        bs2 = brier_score_class(pred2, tgs)

    bss_class = np.ones(bs1.shape) - (bs1 / bs2)
    return bss_class
    
def brier_score_sklearn_weighted(pds, tgs, timestep=None):
    """ Computes Brier Score
        Best score: 0
        Worst score : 1

    Args:
        predictions (np.ndarray): Shape: (samples, timesteps, classes) or (samples, timesteps)
        targets (np.ndarray): Shape: (samples, timesteps)
        timestep (int): Timestep for which to calculate the brier score
    """
    if pds.ndim == 3:
        n_samples, n_timesteps, n_classes = pds.shape
        counts = []
        pdsc = np.argmax(pds, axis = 2)
        for t in range(n_timesteps):
            nums, cts = np.unique(tgs[:,t], return_counts=True)
            if len(cts) < 4:
                class_arr = np.zeros((n_classes,))
                class_arr[nums] = cts
                counts.append(class_arr)
            else:    
                counts.append(cts)
        
    elif pds.ndim == 2:
        n_samples, n_timesteps = pds.shape
        n_classes = np.max(tgs) + 1
    
        counts = []
        for t in range(n_timesteps):
            nums, cts = np.unique(tgs[:,t], return_counts=True)
            if len(cts) < 4:
                class_arr = np.zeros((n_classes,))
                class_arr[nums] = cts
                counts.append(class_arr)
            else:    
                counts.append(cts)
        
        pds = np.eye(n_classes)[pds]
    
    # counts = []
    # for t in range(n_timesteps):
    #     _, cts = np.unique(pds[:,t], return_counts=True)
    #     counts.append(cts)
    
    tgs = np.eye(n_classes)[tgs]
    counts = np.asarray(counts)
    trues = tgs == pds
    counts = np.abs((trues.sum(axis = 0) - counts)) + trues.sum(axis = 0)
    
    norm = (1/(counts))
    
    brier_per_class = []
    for ts in range(n_timesteps):
        for cs in range(n_classes):
            nms = norm[ts,cs]
            y_true = tgs[:,ts,cs].flatten()
            y_pred = pds[:,ts,cs].flatten()
            brier_per_class.append(nms * n_samples * brier_score_loss(y_true, y_pred))
    
    brier_per_class = np.reshape(np.array(brier_per_class),(n_timesteps, n_classes))
    
    brier_per_timestep = np.mean(brier_per_class, axis=1)


    if timestep is not None:
        return brier_per_timestep[timestep]
    else:
        return brier_per_timestep

def classwise_acc(model, test_set, test_loader, trainer):


    pred = []
    for i, batch in enumerate(test_loader): 
        pred.append(model.predict_step(batch, i).detach().numpy())

    predictions = []
    for i in range(len(pred)):
        predictions.append(pred[i])

    targets = []

    for _, output in test_set:
        targets.append(np.array(output).squeeze())
    
    
    scaled_predictions = np.concatenate(predictions)
    n_classes = scaled_predictions.shape[2]
    scaled_classes = np.argmax(scaled_predictions, axis=2)
    ts = scaled_predictions.shape[1]
    targets = np.concatenate(targets).reshape((scaled_predictions.shape[0],ts))

    acc = np.zeros((ts,n_classes))
    for t in range(ts): 
        pdc = scaled_classes[:,t]
        tgt = targets[:,t]
        tgs = np.eye(n_classes)[tgt]
        pds = np.eye(n_classes)[pdc]
        for i in range(n_classes):

            occ_correct = (tgs[:,i].flatten().astype(int) == pds[:,i].flatten().astype(int))* tgs[:,i].astype(int)
            bacc = occ_correct.sum()/tgs[:,i].sum()
            
            acc[t,i] = bacc

    return acc
