FROM datajoint/datajoint

MAINTAINER Fabian Sinz <sinz@bcm.com>

COPY . /server/pipecontrol
WORKDIR /server/pipecontrol

# Update system 
RUN apt update && apt upgrade -y

# Install requirements
RUN apt update && apt install -y graphviz libffi6 libffi-dev libssl-dev
RUN pip3 install --upgrade setuptools # otherwise cairocffi cannot be installed (https://github.com/lief-project/LIEF/issues/108)
RUN pip3 install -r requirements.txt

# Start server
ENTRYPOINT ["./startserver.sh"]
