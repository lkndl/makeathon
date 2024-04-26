
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
Dann funzt
`docker run hello-world`

Und ich folge gerade https://github.com/FeatureCloud/app-tutorial
```
git clone https://github.com/FeatureCloud/app-tutorial
cd app-tutorial
featurecloud app build .
```

Das buildet jetzt zumindest schon mal

