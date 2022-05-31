# Neuron_Burst_Analysis
This project explores various machine learning algorithms and their ability to forecast Local Field Potentials (LFPs).
These ML techniques range from classical shallow models to modern deep learning networks. All model and *in vivo* data
is property of the individual lab and can be found on their OneDrive. 

## Project Outline
In this project, various models can easily be created and run. In the [models folder](lfp_prediction/models) there is a 
list of avaliable models to run and test. Each of these models have a corresponding [config file](configs) that contain
all parameters used to run and test the model. These config files should be the first info looked at when training
a model. A model can be run with a very easy python command using the [lightning runner](lightning_runner.py) script. 
In addition to these models, there is also the ability to [generate and test synthetic data](synthetic%20data%20generator.ipynb).
Finally, if so desired, there is an easy way to [classify samples](Gamma%20Detection.ipynb) as containing gamma bursts 
or not. That being said this project can be broken down into:
1. Classical Models
2. Deep Learning Models

Each using different datasets (synthetic, modeled, and *in vivo*).

## Setup
This project has been run using Python 3.8.10 and the requirements.txt found in the root of this repository.
Simply open your virtual environment and run:
```bash
pip install -r requirements.txt
```

If you are using a conda environment note the hardware running these models had cuda 11.5 installed.

After installing the environment, make sure all paths in the config files (or notebooks you run) point to the correct
data location. This was done on a local machine, so downloading the data is expected. It could benefit from cloud 
storage if there was ever interest in it.

## Classic Models
All classic models can be found in the [Basic TS Models](Basic%20TS%20Models.ipynb) notebook. It contains XGBoost,
SARIMA, and Moving Window. As the majority of these models are simply imported from sklearn, scipy, or stats libraries
they are all contained in one notebook. All that is needed is a valid dataset provided to them.

## Deep Learning Models
All deep models are implemented using [Pytorch](https://pytorch.org/) and [Pytorch Lightning](https://www.pytorchlightning.ai/).
All lightning models also log to [tensorboard](https://www.tensorflow.org/tensorboard). These lightning models are
incredibly easy to run, simply navigate to the [config file](configs) corresponding to the model you want to run, then 
run the command:
```bash
python lightning_runner.py -c configs/<configfilename>
```
**Note:** if an error is thrown saying no logs directory found, simply make a logs directory in the root of the repository.
This is required for tensorboard to log all data.

After finishing this should generate a log file located in the logs directory. This can be viewed in tensorboard using:
```bash
tensorboard --logdir logs/<logdirname>
```
and opening up the link produced on the terminal as one does with jupyter notebook.

To see the prediction simply run the [Result Visualization](Result%20Visualization.ipynb) notebook with the specified 
config and checkpoint. It will all be seeded the same to maintain the correct dataset splits as during training.

## Commentary
The [synthetic generator](synthetic%20data%20generator.ipynb) notebook can function as its own individual machine
learning model notebook or as a way to generate and save synthetic data. The most important part is determining the
location to split intervals. Both Gabor Atoms and Sine Waves can be created in this notebook. The per timestep MSE can 
also be calculated here.
