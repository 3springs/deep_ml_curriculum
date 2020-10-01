# Started on DSVM 18.04 v2 https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-dsvm.ubuntu-1804?tab=Overview


```sh
# start in home dir
cd ~

# essentials on ubuntu, centos used yum
sudo apt install awscli git

# enable nbextensions on base
conda install -y -c conda-forge ipywidgets jupyter_contrib_nbextensions jupyter_nbextensions_configurator
jupyter nbextension enable toc2/main
jupyter nbextension enable toc2/toc2
jupyter nbextension enable execute_time/ExecuteTime
jupyter nbextension enable skip-traceback/main

# get code
git clone https://github.com/3springs/deep_ml_curriculum.git notebooks/deep_ml_curriculum
cd ~/notebooks/deep_ml_curriculum
git checkout run01

# get data (~10Gb)
aws s3 sync s3://deep-ml-curriculum-data/data/processed/ ~/notebooks/deep_ml_curriculum/data/processed/ --region ap-southeast-2 --no-sign-request 

# activate conda env that comes with DSVM 18.04
conda activate py37_pytorch 

# enable nbextensions on py37_pytorch, in case you run jupyter manually from here
conda install -y -c conda-forge ipywidgets jupyter_contrib_nbextensions jupyter_nbextensions_configurator
jupyter nbextension enable toc2/main
jupyter nbextension enable toc2/toc2
jupyter nbextension enable execute_time/ExecuteTime
jupyter nbextension enable skip-traceback/main

# install extra packages from conda main (or as approved)
conda env update --file requirements/environment.min.yml

# extra packages if needed
conda install -c conda-forge umap-learn=0.4.6
pip install xgboost==0.80

# install self
pip install -e .

# install the kernel into jupyter
python -m ipykernel install --user --name py37_pytorch

# to start jupyter, or follow the DSVM instructions https://docs.microsoft.com/en-us/azure/notebooks/use-data-science-virtual-machine

# in jupyter open notebooks/deep_ml_curriculum/notebooks
# choose kernel py37_pytorch

```
