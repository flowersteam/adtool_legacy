FROM nginx:latest
# Setup NGINX reverse proxy
RUN rm /etc/nginx/conf.d/default.conf
COPY services/AutoDiscServer/nginx/nginx.conf /etc/nginx/conf.d/
# Install miniconda and python
RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda install python=3.7 pip
# Install packages for auto_disc lib
WORKDIR /usr/src/libs/auto_disc
COPY libs/auto_disc/setup.py .
RUN pip install -e .
# Install packages for flask app
WORKDIR /usr/src/services/AutoDiscServer/flask_app
COPY services/AutoDiscServer/flask_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Launch Gunicorn as daemon and NGINX
CMD gunicorn -w 1 -b 0.0.0.0:8000 app:app --daemon ; /docker-entrypoint.sh nginx -g "daemon off;"
