FROM datajoint/datajoint

MAINTAINER Fabian Sinz <sinz@bcm.com>

WORKDIR /server/


# Install pipecontrol
COPY . /server/pipecontrol

RUN apt-get update -y && apt-get install -y graphviz python3-pygraphviz

RUN pip3 install -r pipecontrol/requirements.txt

WORKDIR /server/pipecontrol

#ENTRYPOINT gunicorn -b 0.0.0.0:80 manage:app
ENTRYPOINT python3 manage.py runserver

  
