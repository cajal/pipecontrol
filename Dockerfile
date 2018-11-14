FROM datajoint/datajoint

MAINTAINER Fabian Sinz <sinz@bcm.com>

COPY . /server/pipecontrol
WORKDIR /server/pipecontrol

# Install requirements
RUN apt-get update -y && apt-get install -y graphviz libffi6 libffi-dev libssl-dev
RUN pip3 install -r requirements.txt
RUN pip3 install datajoint --upgrade

# Start server
ENTRYPOINT ["./startserver.sh"]
