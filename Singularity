BootStrap: localimage
From: receptor.ants.simg

%post
apt update && apt upgrade -y

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.8 get-pip.py
pip3.8 install numpy==1.23.0
pip3.8 install scipy==1.7.0
pip3.8 install h5py
pip3.8 install cython pandas
pip3.8 install guppy3 cython psutil statsmodels configparser SimpleITK  pyminc  imageio keras pydot  matplotlib nibabel sklearn scikit-image seaborn pykrige torch torchvision antspyx #pandas==0.24.2 webcolors
pip3.8 install stripy
pip3.8 install neurocombat

git clone https://github.com/tfunck/c_solve 
cd c_solve
python3 setup.py install


%environment
LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

%runscript


