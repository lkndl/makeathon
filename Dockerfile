FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc supervisor nginx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./app/server_config/supervisord.conf /supervisord.conf
COPY ./app/server_config/nginx /etc/nginx/sites-available/default
COPY ./app/server_config/docker-entrypoint.sh /entrypoint.sh
COPY requirements.txt /app/requirements.txt

RUN pip3 install --user --upgrade pip
RUN pip3 install --user -r ./app/requirements.txt
RUN pip3 cache purge

COPY ./app /app
COPY ./config.yml /config.yml

EXPOSE 9000 9001
ENTRYPOINT ["sh", "/entrypoint.sh"]
