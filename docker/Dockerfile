
FROM tffunck/brainbuilder.base
MAINTAINER Thomas Funck <tffunck@gmail.com>

# Install PyTorch and torchvision
RUN apt install -y curl python3.8-dev
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py
RUN pip3.8 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html 
RUN git clone https://github.com/pytorch/serve && python3.8 ./serve/ts_scripts/install_dependencies.py --cuda=cu102 && rm -r serve

RUN pip3.8 install torchserve torch-model-archiver torch-workflow-archiver
RUN python3.8 -m pip install --upgrade pip
RUN pip3.8 install antspyx
RUN pip3.8 install  nnunetv2
RUN mkdir -p /opt/ /opt/nnUNet/ /opt/nnUNet/raw /opt/nnUNet/preprocessed /opt/nnUNet/results
ENV nnUNet_raw=/opt/nnUNet/raw
ENV nnUNet_preprocessed=/opt/nnUNet/preprocessed
ENV nnUNet_results=/opt/nnUNet/results
RUN apt install pytest 
RUN apt install -y nvidia-container-toolkit-base
RUN nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
RUN python3.8 	

#pip3.10 install stripy
#pip3.10 install neurocombat
# python3 setup.py install
#RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
#RUN python3.10 get-pip.py
#RUN pip3 install numpy==1.23.0
#RUN pip3 install scipy==1.7.0
#RUN pip3 install cython pandas h5py guppy3 cython psutil statsmodels configparser SimpleITK  pyminc  imageio keras pydot  matplotlib nibabel sklearn scikit-image seaborn pykrige torch torchvision antspyx #pandas==0.24.2 webcolors



