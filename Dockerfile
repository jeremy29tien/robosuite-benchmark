FROM --platform=linux/amd64 continuumio/miniconda3
SHELL ["conda", "run", "/bin/bash", "-c"]
# Install system dependencies
RUN apt update && apt install -y git strace curl vim g++ && rm -rf /var/lib/apt/lists/*
# Set CXX env var 
ENV CXX g++
# Install s5cmd
RUN curl -L https://github.com/peak/s5cmd/releases/download/v2.1.0-beta.1/s5cmd_2.1.0-beta.1_Linux-64bit.tar.gz | tar -xz -C /usr/local/bin && s5cmd --help
# Install python
RUN conda install python=3.7.4

# Install python dependencies for robosuite133
COPY robosuite133/ /code/robosuite133
WORKDIR /code/robosuite133
RUN pip install -e .

# Install python dependencies for robosuite-benchmark
COPY robosuite-benchmark/ /code/robosuite-benchmark
WORKDIR /code/robosuite-benchmark
RUN pip install -r requirements.txt

COPY rlkit/ /code/rlkit
WORKDIR /code/rlkit
RUN pip install -e .

COPY viskit/ /code/viskit
WORKDIR /code/viskit
RUN pip install -e .

# Install pytorch
# RUN conda install python=3.10 pytorch torchvision pytorch-cuda=11.7 -c pytorch-nightly -c nvidia && conda clean -a -y
RUN conda install -c pytorch pytorch

# Expose general port
EXPOSE 3000
# Expose port for jupyter
EXPOSE 8888
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
WORKDIR /code/
RUN export PYTHONPATH=.:$PYTHONPATH

ENTRYPOINT ["/usr/bin/tini", "--"]
