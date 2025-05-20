FROM tffunck/brainbuilder:base

SHELL ["/bin/bash", "-c"]
RUN cd /opt && \
	rm -fr brainbuilder && \
    git clone https://www.github.com/tfunck/brainbuilder &&\
	cd brainbuilder &&\
	/opt/bbenv/bin/pip3 install -r requirements.txt &&\
	/opt/bbenv/bin/pip3 install . 

# Create the entrypoint.sh script
RUN echo '#!/bin/bash' > /opt/entrypoint.sh && \
	echo 'source /opt/bbenv/bin/activate' >> /opt/entrypoint.sh && \
	echo 'exec "$@"' >> /opt/entrypoint.sh

# Make the script executable
RUN chmod +x /opt/entrypoint.sh
ENTRYPOINT ["/opt/entrypoint.sh"]
RUN pwd