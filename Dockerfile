FROM ninai/pipeline:latest

MAINTAINER Fabian Sinz <sinz@bcm.com>

WORKDIR /server/

# get fabee schemata
RUN \
  git clone https://github.com/fabiansinz/fabee.git && \
  git clone https://github.com/fabiansinz/commons.git && \
  git clone https://github.com/fabiansinz/datajoint-addons.git && \
  pip install -e commons/python && \
  pip install -e fabee && \
  pip install -e datajoint-addons

# Install rowbot
COPY . /server/rowbot

RUN \
  pip install -r rowbot/requirements.txt

WORKDIR /server/rowbot

ENTRYPOINT gunicorn -b 0.0.0.0:80 manage:app



  
