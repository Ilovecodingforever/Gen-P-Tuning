"""
load experiments
"""
import time
import wandb
from copy import deepcopy
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model

from train import train
from ts_datasets import InformerDataset, ClassificationDataset

import sys
# sys.path.append('models/moment-research')
# from moment.utils.config import Config
# from moment.utils.utils import control_randomness, parse_config
# from moment.models.gpt4ts_prompt import GPT4TS_prompt
# from moment.models.gpt4ts import GPT4TS

from models.gpt4ts_prompt.utils.config import Config
from models.gpt4ts_prompt.utils.utils import control_randomness, parse_config
from models.gpt4ts_prompt.gpt4ts_prompt import GPT4TS_prompt
from models.gpt4ts_prompt.gpt4ts import GPT4TS

# from momentfm import MOMENTPipeline
from models.moment_prompt import MOMENTPipeline




def load_gpt4ts(task_name: str, forecast_horizon: int,
                n_channels: int, num_classes: int,
                train_loader: torch.utils.data.DataLoader,
                num_prefix: Optional[int] = None,
                multivariate_projection: Optional[str] = None, agg: Optional[str] = None):

        if task_name == 'classification':
            config_path = "src/models/gpt4ts_prompt/configs/prompt/gpt4ts_classification.yaml"
        elif task_name == 'forecasting':
            config_path = "src/models/gpt4ts_prompt/configs/forecasting/gpt4ts_long_horizon.yaml"
        else:
            raise ValueError('task_name must be classification or forecasting')

        gpu_id = 0
        random_seed = 0

        config = Config(
            config_file_path=config_path, default_config_file_path="src/models/gpt4ts_prompt/configs/default.yaml"
        ).parse()

        config["device"] = gpu_id if torch.cuda.is_available() else "cpu"
        args = parse_config(config)
        args.shuffle = False
        args.finetuning_mode = "end-to-end"

        if multivariate_projection is not None:
            args.model_name = "GPT4TS_prompt"
            model = GPT4TS_prompt
        else:
            args.model_name = "GPT4TS"
            model = GPT4TS

        args.n_channels = n_channels
        args.num_prefix = num_prefix
        args.num_class = num_classes
        args.seq_len = train_loader.dataset.seq_len
        args.multivariate_projection = multivariate_projection
        args.agg = agg
        args.forecast_horizon = forecast_horizon

        model = model(configs=args)

        for param in model.parameters():
            param.requires_grad = True

        return model



def prompt(train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
            model_name: str, task_name: str,
            module: str,
            agg: str,
            num_prefix: int = 16,
            epochs: int = 10,
            forecast_horizon: int = 0,
            log=True,):
    """
    prompt tuning
    """

    num_classes = 1
    n_channels = train_loader.dataset.n_channels
    assert task_name in ['classification', 'forecasting']
    if task_name == 'classification':
        num_classes = train_loader.dataset.num_classes

    if model_name == 'moment':
        # model
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                'task_name': task_name,
                'seq_len': train_loader.dataset.seq_len,
                'freeze_encoder': False, # Freeze the patch embedding layer
                'freeze_embedder': False, # Freeze the transformer encoder
                'freeze_head': False, # The linear forecasting head must be trained
                'prefix_tuning_multi': True,
                'forecast_horizon': forecast_horizon,
                'num_prefix': num_prefix,
                'multivariate_projection': module,
                'agg': agg,
                'n_channels': n_channels,
                'num_class': num_classes,
                }
        )
        model.init()

        for n, param in model.named_parameters():
            if 'prefix' not in n and 'prompt' not in n and 'head' not in n and 'value_embedding' not in n and 'layer_norm' not in n:
                param.requires_grad = False

    elif model_name == 'gpt4ts':
        model = load_gpt4ts(task_name, forecast_horizon, n_channels, num_classes, train_loader, num_prefix, module, agg)
        
        for i, (n, param) in enumerate(model.gpt2.named_parameters()):
            if "ln" in n or "wpe" in n or "prompt" in n or "out" in n:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # print not frozen params
    for n, param in model.named_parameters():
        if param.requires_grad:
            print(n)
    print('number of parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))
    start_time = time.time()

    # train
    model = train(model, train_loader, val_loader, test_loader, max_epoch=epochs, 
                    log=log)

    print("--- %s seconds ---" % (time.time() - start_time))

    return model


def no_prompt(train_loader: torch.utils.data.DataLoader,
             val_loader: torch.utils.data.DataLoader,
             test_loader: torch.utils.data.DataLoader,
             model_name: str, task_name: str,
             lora: bool = False, linearprobe: bool = False,
             epochs: int = 10,
             forecast_horizon: int = 0,
             log=True,
             **kwargs):
    """
    finetune
    """

    num_classes = 1
    n_channels = train_loader.dataset.n_channels
    assert task_name in ['classification', 'forecasting']
    if task_name == 'classification':
        num_classes = train_loader.dataset.num_classes

    if model_name == 'moment':
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                'task_name': task_name,
                'seq_len': train_loader.dataset.seq_len,
                'freeze_encoder': False, # Freeze the patch embedding layer
                'freeze_embedder': False, # Freeze the transformer encoder
                'freeze_head': False, # The linear forecasting head must be trained
                'forecast_horizon': forecast_horizon,
                'num_prefix': 16,
                'n_channels': n_channels,
                'num_class': num_classes,
                }
        )
        model.init()

        for param in model.parameters():
            param.requires_grad = True

        if lora:
            model.configs = deepcopy(model.config)
            delattr(model, "config")

            config = LoraConfig(
                r=2,
                lora_alpha=16,
                target_modules=["q", "v"],
                lora_dropout=0.1,
                modules_to_save=["value_embedding", "layer_norm", "head", ],
            )
            model = get_peft_model(model, config)
            model.base_model.model.config = deepcopy(model.configs)


        if linearprobe:
            for n, param in model.named_parameters():
                if 'head' not in n:
                    param.requires_grad = False

    elif model_name == 'gpt4ts':
        model = load_gpt4ts(task_name, forecast_horizon, n_channels, num_classes, train_loader)

        if lora:
            config = LoraConfig(
                r=1,
                lora_alpha=16,
                lora_dropout=0.1,
                # bias="none",
                modules_to_save=["wpe", "enc_embedding", "ln", "predict_linear", "out_layer"],
            )
            model.gpt2 = get_peft_model(model.gpt2, config)

        if linearprobe:
            for n, param in model.named_parameters():
                if 'predict_linear' not in n and 'out_layer' not in n and "enc_embedding" not in n:
                    param.requires_grad = False


    # print not frozen params
    for n, param in model.named_parameters():
        if param.requires_grad:
            print(n)
    print('number of parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))
    start_time = time.time()

    model = train(model, train_loader, val_loader, test_loader, max_epoch=epochs, log=log)

    print("--- %s seconds ---" % (time.time() - start_time))

    return model



def fine_tuning(train_loader: torch.utils.data.DataLoader,
                     val_loader: torch.utils.data.DataLoader,
                     test_loader: torch.utils.data.DataLoader,
                        model_name: str, task_name: str,
                        epochs: int = 10, 
                        forecast_horizon: int = 0,
                        log=True,
                        **kwargs):

    run = wandb.init(
        project="ts-prompt",
    )
    with run:
        return no_prompt(train_loader, val_loader, test_loader, model_name, task_name,
                        epochs=epochs, lora=False, linearprobe=False,
                        forecast_horizon=forecast_horizon, 
                        log=log)



def lora(train_loader: torch.utils.data.DataLoader,
             val_loader: torch.utils.data.DataLoader,
             test_loader: torch.utils.data.DataLoader,
             model_name: str, task_name: str,
             epochs: int = 10,
             forecast_horizon: int = 0,
             log=True,
             **kwargs):

    run = wandb.init(
        project="ts-prompt",
    )

    with run:
        return no_prompt(train_loader, val_loader, test_loader, model_name, task_name,
                        epochs=epochs, lora=True, linearprobe=False,
                        forecast_horizon=forecast_horizon, 
                        log=log)



def linear_probing(train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                model_name: str, task_name: str,
                epochs: int = 10, 
                forecast_horizon: int = 0,
                log=True,
                **kwargs):

    run = wandb.init(
        project="ts-prompt",
    )

    with run:
        return no_prompt(train_loader, val_loader, test_loader, model_name, task_name,
                        epochs=epochs, lora=False, linearprobe=True,
                        forecast_horizon=forecast_horizon, 
                        log=log)




def gen_p_tuning(train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                model_name: str, task_name: str,
                epochs: int = 10, 
                forecast_horizon: int = 0,
                log=True, prompt_size=16,
                **kwargs):

    run = wandb.init(
        project="ts-prompt",
    )

    with run:
        return prompt(train_loader, val_loader, test_loader, model_name, task_name,
                      module='attention', agg='mha',
                        epochs=epochs, 
                        forecast_horizon=forecast_horizon, 
                        num_prefix=prompt_size,
                        log=log)




def prompt_tuning(train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                model_name: str, task_name: str,
                epochs: int = 10, 
                forecast_horizon: int = 0,
                log=True, prompt_size=16,
                **kwargs):
    
    run = wandb.init(
        project="ts-prompt",
    )

    with run:
        return prompt(train_loader, val_loader, test_loader, model_name, task_name,
                      module='vanilla', agg='mha',
                        epochs=epochs, 
                        forecast_horizon=forecast_horizon, 
                        num_prefix=prompt_size,
                        log=log)






