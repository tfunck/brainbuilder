Bootstrap: docker
From: ubuntu:18.04

%post

apt update
apt install -y wget git python3 python3-dev python3-distutils python3-pip ants

pip3 install guppy3 pyminc stipy h5py imageio keras pydot pandas matplotlib nibabel numpy opencv-python sklearn scikit-image scipy seaborn 

%environment
%export LC_ALL=C
%export PATH=/usr/games:$PATH

%runscript


