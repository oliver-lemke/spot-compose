# DeepLearningInit
This is a repository for initializing deep learning projects.

# Setup
Python Version 3.8

<!--
## Conda
```
# create conda environment
conda create --name spot-mask-3d
conda activate spot-mask-3d

sudo apt-get update && sudo apt upgrade -y && sudo apt autoremove
sudo apt-get install -y cdo nco gdal-bin libgdal-dev
python3 -m pip install --upgrade pip setuptools wheel

conda install numpy==1.24.3
# gdalinfo --version, check whether installs
python -m pip install --upgrade gdal==<version>

# install gdal via conda
conda install -c conda-forge libgdal
conda install -c conda-forge gdal
conda install tiledb=2.2
conda install poppler

conda env update -f environment.yml

gdalinfo --version
```

## Installing GDAL
First, install the development headers of libgal-dev
```
sudo apt-get install libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
```

## Installing other dependencies
-->

To install required packages run
```
pip install -r requirements.txt
```

## Overall Setup
Python Version 3.8
<pre>
project_root_dir/                                   <--- root directory of the project
├── source/                                         <--- all code stored here
│   ├── main.py                                     <--- contains the main method
│   ├── trainer.py                                  <--- contains the trainer class responsible for all trainin 
│   │   ├── datasets/
│   │   │   ├── dataset_template.py                 <--- template for how to write a dataset
│   │   │   └── ...
│   ├── models/
│   │   ├── __init__.py                             <--- contains the model_factory which is responsible for building a model
│   │   ├── template_model.py                       <--- template for how a model should look like
│   │   ├── specialized_networks/                   <--- use this folder for special changes to the network
│   │   │   ├── special_example.py                  <--- example for such a network change
│   │   │   └── ...
│   ├── scripts/                                    <--- contains scripts to be run independently (e.g. for setup)
│   │   ├── setup_script.py                         <--- one script do the entire setup, does not do user.yaml config
│   │   └── ...
│   ├── utils/
│   │   ├── configs.py                              <--- ease of use class for accessing config
│   │   ├── eval_metrics.py                         <--- additional metrics to keep track of
│   │   ├── logs.py                                 <--- project-specific logging configuration
│   │   └── ...
│   └── ...
│
├── configs/
│   ├── base.yaml                                   <--- base config file used for changing the actual project
│   ├── template.yaml                               <--- template config for setting up user.yaml
│   └── user.yaml                                   <--- personal config file to set up config for this specific workspace
│
├── data/                                           <--- contains any used datasets
│   ├── README.md                                   <--- markdown file which explains the data and structure
│   └── ...
│
├── logs/                                           <--- contains logs
│   └── ...
│
├── pretrained_weights/                             <--- contains model_weights
│   ├── template_weights/                           <--- template configuration
│   │   ├── weights.pth                             <--- actual weights for the model
│   │   └── pretrained_metadata.pickle              <--- metadata (config used for pretraining)
│
├── output/                                         <--- any model output
│   ├── template_output/
│   │   ├── checkpoints/
│   │   │   ├── weights.pth                         <--- model weights at checkpoint
│   │   │   └── optimizer.pth                       <--- optimizer state at checkpoint
│   │   ├── best_checkpoints/
│   │   └── tensorboard/                            <--- tensorboard directory
│   │   └── wandb/                                  <--- wandb directory
│
├── cache/                                          <--- any local caching that is needed
│   └── ...
│
├── .github/                                        
│   ├── workflows/                                  <--- github actions 
│   │   ├── black.yml
│   │   ├── isort.yml
│   │   ├── pylint.yml
│   │   └── ...
│
├── .gitignore                                      <--- global .gitignore
├── requirements.txt
└── README.md
</pre>

# GitHub Actions
This project uses [black](https://pypi.org/project/black/) and
[isort](https://pypi.org/project/isort/) for formatting, and
[pylint](https://pypi.org/project/pylint/) for linting.

## PyCharm Setup
1. Download the [File Watchers](https://www.jetbrains.com/help/pycharm/using-file-watchers.html)
Plugin
2. Under Settings > Tools > File Watcher > + > \<custom>: setup a new watcher for each
   1. black
      - Name: Black Watcher
      - File type: Python
      - Scope: Project Files
      - Program: $PyInterpreterDirectory$/black
      - Arguments: $FilePath$
      - Output paths to refresh: $FilePath$
      - Working directory: $ProjectFileDir$
      - Additional: as wished
   2. isort
      - Name: iSort Watcher
      - Program: $PyInterpreterDirectory$/isort
      - Arguments: $FilePath$ --sp $ContentRoot$/.style/.isort.cfg --settings-path $ProjectFileDir$/pyproject.toml
   3. pylint
      - Name: PyLint Watcher
      - Program: $PyInterpreterDirectory$/pylint
      - Arguments: --msg-template="$FileDir$/{path}:{line}:{column}:{C}:({symbol}){msg}" $FilePath$ --rcfile $ProjectFileDir$/pyproject.toml
