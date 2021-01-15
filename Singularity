Bootstrap: localimage
From: receptor.base.simg

%post
apt install -y  python3.7 python3.7-dev python3-setuptools
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.7 get-pip.py

pip3.7 install psutil statsmodels configparser stripy SimpleITK scipy numpy pyminc h5py imageio keras pydot pandas==0.24.2 matplotlib nibabel sklearn scikit-image seaborn pykrige guppy

pip3.7 install https://github.com/ANTsX/ANTsPy/releases/download/v0.2.0/antspyx-0.2.0-cp37-cp37m-linux_x86_64.whl
pip3.7 install webcolors


%environment

%runscript


