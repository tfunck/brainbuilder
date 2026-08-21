FROM tffunck/brainbuilder:base

SHELL ["/bin/bash", "-c"]
WORKDIR /opt 
RUN rm -fr brainbuilder && \
    git clone -b dev https://www.github.com/tfunck/brainbuilder
WORKDIR /opt/brainbuilder
RUN  git submodule update --init --recursive
WORKDIR /opt/brainbuilder/morphint
RUN /opt/bbenv/bin/pip3 install . 

RUN /opt/bbenv/bin/pip3 show morphint | grep -i version

WORKDIR /opt/brainbuilder
RUN /opt/bbenv/bin/pip3 install -r requirements.txt 
RUN /opt/bbenv/bin/pip3 install . 


RUN /opt/bbenv/bin/pip3 install --no-cache-dir awscli


# Create the entrypoint.sh script
RUN echo '#!/bin/bash' > /opt/entrypoint.sh && \
	echo 'source /opt/bbenv/bin/activate' >> /opt/entrypoint.sh && \
	echo 'exec "$@"' >> /opt/entrypoint.sh

# Make the script executable
RUN chmod +x /opt/entrypoint.sh
ENTRYPOINT ["/opt/entrypoint.sh"]
RUN pwd
