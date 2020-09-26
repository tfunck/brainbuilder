Bootstrap: docker
From: ubuntu:18.04

%post

apt update
apt install -y gfortran build-essential wget git python3 python3-dev python3-distutils python3-pip ants

pip3 install stripy configparser  SimpleITK scipy numpy guppy3 pyminc h5py imageio keras pydot pandas matplotlib nibabel sklearn scikit-image seaborn 
pip3 install https://github.com/ANTsX/ANTsPy/releases/download/v0.1.4/antspy-0.1.4-cp36-cp36m-linux_x86_64.whl
pip3 install webcolors

%environment
%export LC_ALL=C
%export PATH=/usr/games:$PATH

%runscript


