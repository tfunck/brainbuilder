FROM antsx/ants

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && apt upgrade -y && \
	apt-get install -y \
        git \
		curl\
		python3 python3-dev python3-venv python3-pip\
        python3-opencv \
        libglib2.0-0 \
		build-essential libssl-dev\
		cmake

# Install any python packages you need
WORKDIR /opt

RUN cd /opt && \
	python3 -m venv bbenv &&\
	git clone https://www.github.com/tfunck/brainbuilder && \
	cd brainbuilder && \
	/opt/bbenv/bin/pip3 install -r requirements.txt




