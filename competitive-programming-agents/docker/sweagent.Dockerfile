FROM python:3.11.10-bullseye
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install C++ compiler and build tools
RUN apt-get update && apt-get install -y \
   build-essential \
   cmake \
   g++ \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install swe-rex for faster startup
RUN pip install pipx
RUN pipx install swe-rex
RUN pipx ensurepath
ENV PATH="$PATH:/root/.local/bin/"

# Install any extra dependencies
RUN pip install flake8 scipy

SHELL ["/bin/bash", "-c"]

CMD ["/bin/bash"]