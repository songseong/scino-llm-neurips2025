# SciNO
This repo contains the official implementation for the paper Score-informed Neural Operator for Enhancing Ordering-based Causal Discovery

### Setup Instructions

Run the following command to automatically create a Conda environment, install Python dependencies, and (optionally) install the required R libraries.

```
# Run setup script
bash setup.sh
```

### Training 

```
python main.py --config {config_name}.yml --models {model_name} ...
```

- **{config_name}.yml**: Name of the configuration file to use.
- **{model_name}**: Name of the model to run (choose one: DiffAN or FNO).

**Loading Saved Models**
```
python main.py --config {config_name}.yml --models {model_name} ... --load-ckpt
```

#### Additional Options
- **--CaPS**: Uses the **CaPS** method as the identifiability criterion during causal ordering.
If not specified, the default identifiability criterion is the **DiffAN** method.
- **--probe**: Applies a **probing strategy** based on a pretrained SciNO model by freezing Fourier layers and optimizing only the final MLP.
- **--pretrain**: Performs pretraining followed by probing.

### Probabilistic Control of Autoregressive Causal Ordering
To run LLM-based control for Causal Ordering, use the following command:
```
python main_control.py 
```
Make sure to specify the target dataset by setting the dataset_name field in the configuration file:
**configs/control/_control.yml**

### Pretrained checkpoints

All **checkpoints** are provided in this [Google Drive link](https://drive.google.com/file/d/1ZZ5godXYGNQGVLrFNJ5mtx0CCTlaaT2_/view?usp=drive_link).

### Dataset

All **datasets** are provided in this [Google Drive link](https://drive.google.com/file/d/1GgucMenfLw_jlZRg7vLsoJcSMTW8sAZ1/view?usp=drive_link).