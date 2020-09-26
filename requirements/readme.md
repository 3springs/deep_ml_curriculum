# Started on DSVM 18.04 v2 https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-dsvm.ubuntu-1804?tab=Overview

# start in home dir
cd ~

# essentials on ubuntu, centos used yum
sudo apt install awscli git

# get code
git clone https://github.com/3springs/deep_ml_curriculum.git notebooks/
cd ~/notebooks/deep_ml_curriculum

# get data (~10Gb)
aws s3 sync s3://deep-ml-curriculum-data/data/processed/ ~/notebooks/deep_ml_curriculum/data/processed/ --region ap-southeast-2 --no-sign-request 

# activate conda env that comes with DSVM 18.04
conda activate py37_pytorch

# install extra packages from conda main (or as approved)
conda env update --file requirements/environment.min.yml

# extra packages
conda install -c conda-forge umap-learn=0.4.6
pip install xgboost==0.80
