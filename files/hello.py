from dataclasses import dataclass

@dataclass
class IMTSForecastingConfig:
    state: str = 'def'
    n: int = int(1e8)  # Size of the dataset
    hop: int = 1  # hops in GNN
    nhead: int = 1  # heads in Transformer
    tf_layer: int = 1  # # of layer in Transformer
    nlayer: int = 1  # # of layer in TSmodel
    epoch: int = 1000  # training epoches
    patience: int = 10  # patience for early stop
    history: int = 24  # number of hours (months for ushcn and ms for activity) as historical window
    patch_size: float = 24  # window size for a patch
    stride: float = 24  # period stride for patch sliding
    logmode: str = "a"  # File mode of logging.

    lr: float = 1e-3  # Starting learning rate.
    w_decay: float = 0.0  # weight decay.
    batch_size: int = 32

    save: str = 'experiments/'  # Path for save checkpoints
    load: str | None = None  # ID of the experiment to load for evaluation. If None, run a new experiment.
    seed: int = 1  # Random seed
    dataset: str = 'physionet'  # Dataset to load. Available: physionet, mimic, ushcn

    # value 0 means using original time granularity, Value 1 means quantization by 1 hour, 
    # value 0.1 means quantization by 0.1 hour = 6 min, value 0.016 means quantization by 0.016 hour = 1 min
    quantization: float = 0.0
    model: str = 'tPatchGNN'  # Model name
    outlayer: str = 'Linear'  # Model name
    hid_dim: int = 64  # Number of units per hidden layer
    te_dim: int = 10  # Number of units for time encoding
    node_dim: int = 10  # Number of units for node vectors
    gpu: str = '0'  # which gpu to use.
