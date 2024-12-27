FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip3 install tqdm numpy

COPY POINTNET /app/
COPY POINTNET-2048 /app/
COPY POINTNET-atomchannel /app/
COPY prepare_data.py /app/
RUN chmod +x POINTNET POINTNET-2048 POINTNET-atomchannel

ENTRYPOINT ["python3"] 