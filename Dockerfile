FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN adduser --shell /bin/bash -uid 1002 app
USER app
COPY --chown=app:app requirements.txt requirements.txt

USER root
RUN pip3 install -U pip
RUN pip3 install -r requirements.txt

USER app
COPY --chown=app:app . .

EXPOSE 8502

HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health

ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8502", "--server.address=0.0.0.0"]
