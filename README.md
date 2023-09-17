# Training

```bash
python3 train.py -pm demo-data/model-p1.hdf5 -pn demo-net -e 120 -k 3 -bs 1024
```

```bash
usage: train.py [-h] [-pm PATH_FILEMODEL] [-pn PATH_NETOUT] [-e EPOCH] 
                     [-bs BATCH_SIZE] [-k KFOLD] [-o OPTIMIZER] [-l LOSS]

options:
  -h, --help            show this help message and exit
  -pm PATH_FILEMODEL, --path_filemodel PATH_FILEMODEL
                        Select the smash Model object
  -pn PATH_NETOUT, --path_netout PATH_NETOUT
                        [optional] Select the output directory for the trained neural network
  -e EPOCH, --epoch EPOCH
                        [optional] Select the number of epochs for training
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        [optional] Select the batch size for training
  -k KFOLD, --kfold KFOLD
                        [optional] Select the number of folds for cross-validation
  -o OPTIMIZER, --optimizer OPTIMIZER
                        [optional] Select the optimization algorithm
  -l LOSS, --loss LOSS  [optional] Select the loss function for optimization
```

# Prediction

```bash
python3 predict.py -pm demo-data/model-p2.hdf5 -pn demo-net -po bias-p2.csv
```

```bash
usage: predict.py [-h] [-pm PATH_FILEMODEL] [-pn PATH_NET] [-po PATH_FILEOUT] [-bs BATCH_SIZE]

options:
  -h, --help            show this help message and exit
  -pm PATH_FILEMODEL, --path_filemodel PATH_FILEMODEL
                        Select the smash Model object to correct
  -pn PATH_NET, --path_net PATH_NET
                        Select the trained neural network to correct the Model object
  -po PATH_FILEOUT, --path_fileout PATH_FILEOUT
                        [optional] Select path for the output file
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        [optional] Select the batch size for predicting
```

# Validation Results

```python
import smash
import pandas as pd

def correct_bias(model: smash.Model, df: pd.DataFrame):
    model_correct = model.copy()

    for i, code in enumerate(model.mesh.code):
        if code in df.columns[1:]:
            model_correct.response.q[i, df["timestep"].values] += df[code].values

    return model_correct

df = pd.read_csv("bias-p2.csv")
model = smash.io.read_model("demo-data/model-p2.hdf5")
model_correct = correct_bias(model, df)
```

```python
print("Original model KGE: ", smash.metrics(model, "kge"))
print("Corrected model KGE: ", smash.metrics(model_correct, "kge"))
```
```bash
Original model KGE:  [0.72716784 0.38292903 0.68023837 0.59912944]
Corrected model KGE:  [0.85017586 0.65304232 0.87735885 0.75716603]
```