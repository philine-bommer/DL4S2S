import os
import yaml
import pickle
import json
from argparse import ArgumentParser
import pdb

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.utils import shuffle
from sklearn.datasets import make_classification
from sklearn.utils import class_weight



from dataset.datasets import PlainData
from utils import concat_data, statics_from_config
from utils_evaluation import balanced_accuracy

if __name__ == '__main__':

    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--config", type=str, default='')
    parser.add_argument("--ntrials", type=int, default=100)

    args = parser.parse_args()
    num_mods = args.ntrials

    seeds = np.arange(1,num_mods +1, dtype = int)

    cfile = args.config

    # Load config and settings.
    cfd = os.path.dirname(os.path.abspath(__file__))
    config = yaml.load(open(f'{cfd}/config/convlstm_config{cfile}.yaml'), Loader=yaml.FullLoader)

    strt_yr = config.get('strt','')
    trial_num = config.get('version', '')
    target_dir = config['lr_dir'] + f'_{strt_yr}{trial_num}/statistics/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    var_comb = config['var_comb']

    data_info, _ = statics_from_config(config)

    seasons =  {'train':{config['data']['dataset_name2']:list(range(config['data']['fine']['train_start'], config['data']['fine']['train_end']))},
        'val':{config['data']['dataset_name2']:list(range(config['data']['fine']['val_start'], config['data']['fine']['val_end']))},
        'test':{config['data']['dataset_name2']:list(range(config['data']['fine']['test_start'], config['data']['fine']['test_end']))}}


    # Create data loader.
    params = {'seasons': seasons, 'test_set_name':config['data'][config['name']]['setname']}
    Pre_data = PlainData(config['data']['dataset_name1'], 
                                var_comb, data_info, config['data']['bs'], **params)
    Fine_data = PlainData(config['data']['dataset_name2'], 
                                var_comb, data_info, config['data']['bs'], **params)

    if config['transfer']:
        X_train, y_train, _, _, X_test, y_test = concat_data(Pre_data, Fine_data)
    else:
        X_train, y_train = Fine_data.train_data()
        X_val, y_val = Fine_data.val_data()
        X_test, y_test =Fine_data.test_data()
   

    Y_train = y_train#np.argmax(y_train, axis = -1)
    Y_test = y_test#np.argmax(y_test, axis = -1)

    cls_wt = class_weight.compute_class_weight('balanced', 
                            classes = np.unique(y_train), 
                            y = y_train.flatten())
    
  
    input_dim = X_train.shape[1]
    # remember for reproducibility and analysis
    data_info['var_comb'] = var_comb


    # Build model
    model_params = dict(multi_class='multinomial', solver='lbfgs', max_iter=1000,
                        tol=0.0001, verbose=0, class_weight = 'balanced')
    
    experiment_info = {
        'data_info': data_info,
        'model_parameters': model_params
    }

    with open(os.path.join(target_dir, 'model_architecure.json'), 'w+') as f:
        json.dump(experiment_info, f, indent=4)
    
    acc_ts = []
    for i in seeds:
        np.random.seed(i)

        # Set up models args.
        model = LogisticRegression(**model_params, random_state = i)


        # Run Training.
        model.fit(X_train, Y_train)

        acc = model.score(X_test, Y_test)

        test_probs = model.predict_proba(X_test)
        test_preds = model.predict(X_test)


        test_probs = test_probs.reshape(int(test_probs.shape[0]/6),6,
                                        test_probs.shape[1])
        test_preds =test_preds.reshape(int(test_preds.shape[0]/6),6)
        targets = Y_test.reshape(int(Y_test.shape[0]/6),6)

        overall_accuracy_lr, bas_lr = balanced_accuracy(np.argmax(test_probs, axis=-1), targets)

        acc_ts.append(overall_accuracy_lr)
    pdb.set_trace()
    accuracy_ts = np.concatenate(acc_ts).reshape(num_mods,6)

    print(f'Accuracy mean: {accuracy_ts.mean(axis=0)}, var: {accuracy_ts.var(axis=0)}')
