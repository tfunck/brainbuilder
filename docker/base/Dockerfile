FROM antsx/ants
MAINTAINER Thomas Funck <tffunck@gmail.com>

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && apt upgrade -y && \
	apt-get install -y \
        git \
	curl\
	python3.8 python3.8-dev\
        python3-opencv \
        libglib2.0-0 \
	build-essential libssl-dev\
	cmake \
        python3-pip 

RUN apt install -y curl python3.8-dev
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py
RUN pip3.8 install --upgrade pip
# Install any python packages you need
COPY requirements.txt requirements.txt
RUN python3.8 -m pip install -r requirements.txt
