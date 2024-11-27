import os
import random
from argparse import Namespace
from typing import NamedTuple

import numpy as np
import torch


def control_randomness(seed: int = 13):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    