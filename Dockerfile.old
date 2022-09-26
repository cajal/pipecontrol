FROM datajoint/datajoint:v0.12.9

MAINTAINER Fabian Sinz <sinz@bcm.com>, Cameron Smith <camerons@bcm.edu>, Christos Papadopoulos <cpapadop@bcm.edu>

COPY . /server/pipecontrol
WORKDIR /server/pipecontrol

# Update system 
USER root
RUN apt-get update && apt-get upgrade -y

# Install base requirements
RUN apt-get install -y graphviz libffi6 libffi-dev libssl-dev cron

# USER 1001

# Might want to install python3.8 and point everything below to that

# Install python requirements
RUN python3 -m pip install -U pip
RUN pip3 install --upgrade setuptools # otherwise cairocffi cannot be installed (https://github.com/lief-project/LIEF/issues/108)
RUN pip3 install --upgrade git+https://github.com/atlab/datajoint-python # fixes reconnection bug without losing compatibility
RUN pip3 install -r requirements.txt

# Start server
ENTRYPOINT ["./startserver.sh"]
