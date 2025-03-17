import argparse
import os
import yaml
import pdb
from pathlib import Path

import lightning.pytorch as pl
import optuna

from hyperparameter_tuning_objective import objective_vit


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    parser.add_argument("--config", type=str, default='')
    parser.add_argument("--network", type=str, default='ViT')
    parser.add_argument("--ntrials", type=int, default=100)
    args = parser.parse_args()

    cfile = args.config
    ntype = args.network
    ntr = args.ntrials
    # Load config and settings.
    exd = os.path.dirname(os.path.abspath(__file__))
    cfd = exd.parent.absolute()

    config = yaml.load(open(f'{cfd}/config/loop_config{cfile}.yaml'), Loader=yaml.FullLoader)
    arch = 'ViT-LSTM/'

    config['net_root'] = str(cfd.parent.absolute()) + f'/Data/Network/'
    config['data_root'] = str(cfd.parent.absolute()) + f'/Data'

    strt_yr = config.get('strt','')
    trial_num = config.get('version', '')
    norm_opt = config.get('norm_opt','')
    name_var = config.get('tropics','')

    results_directory = config['net_root'] + f'Sweeps/{arch}Sweep_{strt_yr}{trial_num}_{norm_opt}{name_var}/'
    os.makedirs(Path(results_directory), exist_ok = True) 

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)

    study.optimize(objective_vit, n_trials=ntr) 
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    param_list = []
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        param_list.append(key)



    num = trial.number

    results = {'folder': f"version_{num}", 'val_acc - val_ece': trial.value, 'Params':trial.params}
    with open(results_directory + '/results.yml', 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False)

    
    with open(results_directory + '/config.yml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


    # Plot Visulizations.
    # plot_dir = results_directory + "/Plots/"   
    # os.makedirs(Path(plot_dir), exist_ok = True) 
        
    # fig1 = optuna.visualization.plot_intermediate_values(study)
    # fig1.write_image(f"{plot_dir}/intermediate_results.png") 

    # fig2 = optuna.visualization.plot_optimization_history(study)
    # fig2.write_image(f"{plot_dir}/optimization_history.png")


    # fig3 = optuna.visualization.plot_parallel_coordinate(study, params=param_list)
    # fig3.write_image(f"{plot_dir}/parallel_coordinate.png")

    # fig4 = optuna.visualization.plot_param_importances(study)
    # fig4.write_image(f"{plot_dir}/parameter_importance.png")

