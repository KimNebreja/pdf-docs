release: apt-get update && apt-get install -y pandoc
web: gunicorn -b 0.0.0.0:$PORT pdf-docs-api:app
