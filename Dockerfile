FROM datajoint/datajoint

MAINTAINER Fabian Sinz <sinz@bcm.com>, Cameron Smith <camerons@bcm.edu>

COPY . /server/pipecontrol
WORKDIR /server/pipecontrol

# Update system 
RUN apt update && apt upgrade -y

# Install requirements
RUN apt update && apt install -y graphviz libffi6 libffi-dev libssl-dev cron
RUN pip3 install --upgrade setuptools # otherwise cairocffi cannot be installed (https://github.com/lief-project/LIEF/issues/108)
RUN pip3 install --upgrade git+https://github.com/atlab/datajoint-python # fixes reconnection bug without losing compatibility
RUN pip3 install -r requirements.txt

# Start server
ENTRYPOINT ["./startserver.sh"]
