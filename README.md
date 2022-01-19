# Text-Classification-with-Transformers


# Installing Environments
First create conda environment with "condo create --name envname python=3.8.10 pip"

Then run: "pip install -r requirements_pip_new.txt" #installs almost all needed packages in the environment

Then run

pip install "git+https://github.com/google-research/robustness_metrics.git#egg=robustness_metrics" #installs package from GitHub page

conda install --channel conda-forge nb_conda_kernels #install nb_conda_kernels to make kernels available in environments from conda-forge channel

conda install cudatoolkit=10.1.243 cudnn=7.6.5 #GPU Support, cudatoolkit & cudnn version depends on your GPU & installed drivers. If you only want to use it with CPU (i.e. for predictions) then you can ignore this step.


# 'Manual'
The Training Notebook lets one train text classification models.

The Classification with Reject Option Evaluation Notebook lets one evaluate trained models regarding classification with reject option and uncertainty calibration.

The Timing Script can be used to get an indication of runtimes of the different methods used in classification with reject option.

The .py scripts are loaded into the notebooks that serve as a sort easy-to-use interface.

Structure of the files are specific to a master thesis which was created using the scripts. If one wants to apply the scripts to other text classification tasks, adjustments may be necessary (e.g. different languages, maximum sequence length)


