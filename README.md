
## wie man eine featurecloud app lokal startet - Versuche

Installiere ein Python, dann das FeatureCloud package:
`pip install featurecloud`
oder


Installiere docker und starte einen daemon, siehe [hier](https://docs.docker.com/config/daemon/start/). Auf Linux/WSL mit
`sudo systemctl start docker`
oder mit `restart`.

Dann gibt es denn Test-Befehl
`docker run hello-world`
Wenn man so einen `permission denied` kriegt ist [hier](https://stackoverflow.com/questions/48957195/how-to-fix-docker-got-permission-denied-issue) eine Lösung und [dort](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user) die offizielle Handbuchseite dazu.
Dann funzt `docker run hello-world` oder `docker info` ohne `permission denied`! Bei mir war der Neustart wirklich hilfreich; auch auf linux. Do it!

Und ich folge gerade https://github.com/FeatureCloud/app-tutorial
```
git clone https://github.com/FeatureCloud/app-tutorial
cd app-tutorial
featurecloud app build . bionerds
```
Das buildet vielleicht erfolgreich. Dann man muss jetzt einen controller starten.
`featurecloud controller start`
Theoretisch können mehrere laufen (mit `ls` statt `start` listen) aber bei mir ging das nicht. Denke in dem Bsp ist irgendwas hardgecodet, das sich da quer stellt. Man sieht der controller aber auf `http://localhost:8000/`

Mit `docker images` könnt ihr nachschauen welche images ihr habt, und dann zB `bionerds` hiermit starten:

`featurecloud test start --app-image bionerds`

# FeatureCloud Testing

Das habe ich von den featurecloud [dev docs](https://featurecloud.ai/assets/developer_documentation/getting_started.html)

Dabei wird euch eine Test-ID angezeigt! Mit `featurecloud test info --test-id 5` sieht man bissl was, und in `/data/tests` findet ihr wahrscheinlich Ergebnisse davon. Bei mir sind das für das `mnist`-Repo leere zip files
Legt euch einen featurecloud.ai-Account an, dann könnt ihr euch das fancy im browser anschauen: https://featurecloud.ai/development/test

Für's deployment braucht man auch noch einen Docker-Account. Dann macht man 
```
docker login
docker login featurecloud.ai
# Dann die App bionerds auf featurecloud.ai im Browser anlegen! Erst dann funktioniert:
featurecloud app publish bioners
```

# PersonalizedMedicine container Docker Testing

Full restart and rebuild
```bash
sudo systemctl restart docker
make build
docker run -d -v ./config.yml:/mnt/input/config.yml -v ./data/output:/mnt/output -p 9000:9000 featurecloud.ai/bionerds:latest
curl --location 'http://localhost:9000/setup' --header 'Content-Type: application/json' --data '{"id": "0000000000000000","coordinator": false,"coordinatorID": "0000000000000000","clients": []}'
docker container ls && CONTAINER_ID=$(docker container ls | tail -n 1 | cut -d ' ' -f 1) && echo "CONTAINER_ID is right now: $CONTAINER_ID"
docker logs $CONTAINER_ID
```

If you're building and re-building right now: Make changes, then:
```bash
make build
# get the old container ID first:
docker container ls && CONTAINER_ID=$(docker container ls | tail -n 1 | cut -d ' ' -f 1) && echo "CONTAINER_ID is right now: $CONTAINER_ID"
docker stop $CONTAINER_ID
# start the new version
docker run -d -v ./config.yml:/mnt/input/config.yml -v ./data/output:/mnt/output -p 9000:9000 featurecloud.ai/bionerds:latest
docker container ls && CONTAINER_ID=$(docker container ls | tail -n 1 | cut -d ' ' -f 1) && echo "CONTAINER_ID just changed and is now: $CONTAINER_ID"
curl --location 'http://localhost:9000/setup' --header 'Content-Type: application/json' --data '{"id": "0000000000000000","coordinator": false,"coordinatorID": "0000000000000000","clients": []}'

# This is the output
docker logs $CONTAINER_ID
```