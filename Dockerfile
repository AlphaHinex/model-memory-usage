FROM python:3.8-slim

RUN app update
RUN apt-get install -y --no-install-recommends build-essential git gcc

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
WORKDIR /workspace

RUN python3 -m pip install --upgrade --no-cache-dir pip
# Install base dependencies:
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install no-dep packages
RUN python3 -m pip install --no-cache-dir --no-deps -r no_dep_requirements.txt

RUN app.py