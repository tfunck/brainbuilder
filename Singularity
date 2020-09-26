Bootstrap: docker
From: ubuntu:18.04

%post

apt update
apt install -y gfortran build-essential wget git python3 python3-dev python3-distutils python3-pip ants

pip3 install stripy scipy numpy guppy3 pyminc h5py imageio keras pydot pandas matplotlib nibabel sklearn scikit-image seaborn 

%environment
%export LC_ALL=C
%export PATH=/usr/games:$PATH

%runscript


