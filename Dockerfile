FROM --platform=linux/amd64 continuumio/miniconda3

# TEMPORARY:
RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf

SHELL ["conda", "run", "/bin/bash", "-c"]
# Install system dependencies
RUN apt update && apt install -y git strace curl vim g++ && rm -rf /var/lib/apt/lists/*
# Set CXX env var 
ENV CXX g++
# Install s5cmd
RUN curl -L https://github.com/peak/s5cmd/releases/download/v2.1.0-beta.1/s5cmd_2.1.0-beta.1_Linux-64bit.tar.gz | tar -xz -C /usr/local/bin && s5cmd --help
# Install python
RUN conda install python=3.8.0

# Install python dependencies for robosuite133
COPY robosuite133/ /code/nl_pref/robosuite133
WORKDIR /code/nl_pref/robosuite133/
RUN pip install -e .

# Install python dependencies for robosuite-benchmark
COPY robosuite-benchmark/ /code/nl_pref/robosuite-benchmark
WORKDIR /code/nl_pref/robosuite-benchmark/
RUN pip install -r requirements.txt

COPY rlkit/ /code/nl_pref/rlkit
WORKDIR /code/nl_pref/rlkit/
RUN pip install -e .

COPY viskit/ /code/nl_pref/viskit
WORKDIR /code/nl_pref/viskit/
RUN pip install -e .

# Install pytorch and other dependencies
# TODO: may need to update pytorch to latest version
RUN conda install pytorch nomkl
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev
RUN pip install patchelf

# Set up stuff for mujoco
WORKDIR /code/nl_pref/
COPY .mujoco/ /code/nl_pref/.mujoco
RUN cp -R .mujoco/ /root/
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin' >> /root/.bashrc

# Modify pythonpath for robosuite-benchmark
WORKDIR /code/nl_pref/robosuite-benchmark/
#RUN export PYTHONPATH=.:$PYTHONPATH
ENV PYTHONPATH=.:$PYTHONPATH

# Expose general port
EXPOSE 3000
# Expose port for jupyter
EXPOSE 8888
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
WORKDIR /code/
ENTRYPOINT ["/usr/bin/tini", "--"]

WORKDIR /code/nl_pref/robosuite-benchmark/
