Bootstrap: docker
From: ubuntu:18.04

%post
wget https://bootstrap.pypa.io/get-pip.py
RUN python2.7 get-pip.py
RUN pip2.7  install networkx nipype nibabel pydot h5py numpy scipy configparser pandas nibabel weave sklearn seaborn


%environment
%export LC_ALL=C
%export PATH=/usr/games:$PATH

%runscript
%fortune | cowsay | lolcat
