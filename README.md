# Text-Classification-with-Transformers


# Installing Environments
First create conda environment with "condo create --name envname python=3.8.10 pip"

Then run: "pip install -r requirements_pip_new.txt" #installs almost all needed packages in the environment

Then run

pip install "git+https://github.com/google-research/robustness_metrics.git#egg=robustness_metrics" #installs package from GitHub page
conda install --channel conda-forge nb_conda_kernels #install nb_conda_kernels to make kernels available in environments from conda-forge channel
conda install cudatoolkit=10.1.243 cudnn=7.6.5 #GPU Support, cudatoolkit & cudnn version depends on your GPU & installed drivers. If you only want to use it with CPU (i.e. for predictions) then you can ignore this step.
