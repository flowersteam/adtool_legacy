FROM python:3.7.14-slim

# resolve project dependencies
COPY ./pyproject.toml /usr/lib/auto_disc/pyproject.toml
RUN pip install -e .

# copy source code
COPY . /usr/lib/auto_disc
WORKDIR /usr/lib/auto_disc
