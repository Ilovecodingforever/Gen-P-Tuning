import torch.cuda.amp

import sys

import os
import torch
import datetime


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


RANDOM_SEED = 13


from utils import control_randomness




# Set random seeds for PyTorch, Numpy etc.
control_randomness(seed=RANDOM_SEED)


def classify_experiments(experiment_name: str,
                         model_name: str,
                         epochs: int = 10,
                         prompt_size: int = 16,
                         seed=0,):

    from data import get_data
    from experiments import gen_p_tuning, prompt_tuning, linear_probing, lora, fine_tuning

    task = 'classification'
    batch_size = 1

    if experiment_name == 'gen_p_tuning':
        experiment = gen_p_tuning
    elif experiment_name == 'fine_tuning':
        experiment = fine_tuning
    elif experiment_name == 'lora':
        experiment = lora
    elif experiment_name == 'linear_probing':
        experiment = linear_probing
    elif experiment_name == 'prompt_tuning':
        experiment = prompt_tuning
    else:
        raise NotImplementedError('experiment not supported')

    # from gpt4ts
    experiment_files = {
        'EthanolConcentration': "../data/Timeseries-PILE/classification/UCR/EthanolConcentration/EthanolConcentration",
        'JapaneseVowels': "../data/Timeseries-PILE/classification/UCR/JapaneseVowels/JapaneseVowels",
        'SelfRegulationSCP1': "../data/Timeseries-PILE/classification/UCR/SelfRegulationSCP1/SelfRegulationSCP1",
        'SelfRegulationSCP2': "../data/Timeseries-PILE/classification/UCR/SelfRegulationSCP2/SelfRegulationSCP2",
        'UWaveGestureLibrary': "../data/Timeseries-PILE/classification/UCR/UWaveGestureLibrary/UWaveGestureLibrary",
        'SpokenArabicDigits': "../data/Timeseries-PILE/classification/UCR/SpokenArabicDigits/SpokenArabicDigits",
    }

    # check file exist
    assert all([os.path.exists(file+'_TEST.ts') for file in experiment_files.values()])


    for dataset_name, filename in experiment_files.items():

        control_randomness(seed=13) # make sure same as moment
        train_loader, val_loader, test_loader = get_data(batch_size=batch_size, task=task, filename=filename)

        control_randomness(seed=seed)
        _ = experiment(train_loader, val_loader, test_loader,
                        model_name, task,
                        epochs=epochs,
                        prompt_size=prompt_size,
                        )




def long_forecast_experiments(experiment_name: str,
                              model_name: str,
                              epochs: int = 10,
                              prompt_size: int = 16,
                              seed=0,
                              ):

    from data import get_data
    from experiments import gen_p_tuning, prompt_tuning, linear_probing, lora, fine_tuning

    task = 'forecasting'
    batch_size = 1
    # batch_size = 64

    if experiment_name == 'gen_p_tuning':
        experiment = gen_p_tuning
    elif experiment_name == 'fine_tuning':
        experiment = fine_tuning
    elif experiment_name == 'lora':
        experiment = lora
    elif experiment_name == 'linear_probing':
        experiment = linear_probing
    elif experiment_name == 'prompt_tuning':
        experiment = prompt_tuning
    else:
        raise NotImplementedError('experiment not supported')


    experiment_files = {
        'national_illness': ("data/Timeseries-PILE/forecasting/autoformer/national_illness.csv",
                             [60]),
        # 'exchange_rate': ("data/Timeseries-PILE/forecasting/autoformer/exchange_rate.csv", [96]),
        # 'ETTh1': ("data/Timeseries-PILE/forecasting/autoformer/ETTh1.csv", [96]),
        # 'ETTh2': ("data/Timeseries-PILE/forecasting/autoformer/ETTh2.csv", [96]),
    }
    # check file exist
    assert all([os.path.exists(file) for file, _ in experiment_files.values()])


    for dataset_name, (filename, horizons) in experiment_files.items():
        for horizon in horizons:

            control_randomness(seed=13) # make sure same as moment
            train_loader, val_loader, test_loader = get_data(batch_size=batch_size, task=task,
                                                                filename=filename, forecast_horizon=horizon,)

            control_randomness(seed=seed)
            _ = experiment(train_loader, val_loader, test_loader, 
                           model_name, task,
                            epochs=epochs,
                            forecast_horizon=horizon,
                            prompt_size=prompt_size, 
                            )




def mimic_experiments(experiment_name: str,
                         model_name: str,
                         benchmark: str,
                         epochs: int = 10,
                         prompt_size: int = 16,
                         seed=0,
                         ):

    from data import load_mimic
    from experiments import gen_p_tuning, prompt_tuning, linear_probing, lora, fine_tuning

    batch_size = 1
    task = 'classification'

    if experiment_name == 'gen_p_tuning':
        experiment = gen_p_tuning
    elif experiment_name == 'fine_tuning':
        experiment = fine_tuning
    elif experiment_name == 'lora':
        experiment = lora
    elif experiment_name == 'linear_probing':
        experiment = linear_probing
    elif experiment_name == 'prompt_tuning':
        experiment = prompt_tuning
    else:
        raise NotImplementedError('experiment not supported')


    control_randomness(seed=seed)

    train_loader, val_loader, test_loader = load_mimic(benchmark=benchmark, seed=seed, batch_size=batch_size,)

    _ = experiment(train_loader, val_loader, test_loader, 
                   model_name, task,
                    epochs=epochs,
                    prompt_size=prompt_size,
                    )






if __name__ == "__main__":

    model = 'moment'
    # model = 'gpt4ts'

    experiment = 'gen_p_tuning'
    # experiment = 'fine_tuning'
    # experiment = 'lora'
    # experiment = 'linear_probing'
    # experiment = 'prompt_tuning'


    prompt_size = 16
    # classify_experiments(experiment, model, prompt_size=prompt_size) 
    long_forecast_experiments(experiment, model, prompt_size=prompt_size) 


    prompt_size = 4
    # benchmark = 'mortality'
    benchmark = 'phenotyping'
    # mimic_experiments(experiment, model, benchmark, prompt_size=prompt_size)


