Bootstrap: docker
From: ubuntu:18.04

%post

apt update

apt install -y git 

cd /opt

git clone https://40dfc4ecd9e2e820e5c52dd694805956e95a8b83@github.com/tfunck/julich-receptor-atlas 

apt install -y tensorflow vim cmake python3 python3-dev python3-distutils python3-pip libsm6 libxrender-dev

pip3 install  h5py imageio keras pandas matplotlib nibabel numpy opencv-python sklearn scikit-image scipy seaborn statsmodels 

git clone https://github.com/ANTsX/ANTsPy
cd ANTsPy
python3 setup.py install
cd ..
rm -r ANTsPy

%environment
%export LC_ALL=C
%export PATH=/usr/games:$PATH

%runscript

