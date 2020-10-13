# FROM continuumio/miniconda3:latest
# ### IF GPU USE BELOW IMAGE
FROM nvidia/cuda:10.2-base
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update && \
    apt-get install -y wget && rm -rf /var/lib/apt/lists/* && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh && \
    conda init bash && \
    exec bash && \
    conda activate base
### PREP
# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]
RUN apt-get update -y
### TESSERACT
ARG version=4.1.1
WORKDIR /usr/local/share/tessdata/
RUN wget https://github.com/tesseract-ocr/tessdata/raw/master/eng.traineddata && \
    wget https://github.com/tesseract-ocr/tessdata/raw/master/osd.traineddata && \
    wget https://github.com/tesseract-ocr/tesseract/archive/$version.zip && \
    wget -O eng_best.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/master/eng.traineddata && \
    apt-get install -y unzip && \
    unzip $version.zip
WORKDIR /usr/local/share/tessdata/tesseract-$version
RUN apt-get install -y libleptonica-dev && \
    apt-get install -y automake libtool g++ make && \
    apt-get install -y pkg-config && \
    apt-get install -y libsdl-pango-dev && \
    apt-get install -y libicu-dev && \
    apt-get install -y libcairo2-dev && \
    apt-get install -y bc && \
    ./autogen.sh && \
    ./configure --disable-dependency-tracking && \
    make && \
    make install && \
    ldconfig && \
    make training && \
    make training-install
# ### PYTHON ENVIRONMENT
WORKDIR /code
COPY . .
# libgl1-mesa-glx fixes opencv error: https://github.com/ContinuumIO/docker-images/issues/49
# RUN apt-get install libgl1-mesa-glx unzip -y && \
#     conda env create -f conda_environment.yml && \
#     echo "conda activate ml-server" > ~/.bashrc && \
#     conda init bash && \
#     conda activate ml-server

EXPOSE 5000

# # ENTRYPOINT ["conda", "run", "-n", "ml-server", "python", "benchmark.py"]

# # ENTRYPOINT [ \
# #     "conda", "run", "-n", "ml-server", "python", "benchmark.py", \
# #     "&&", "conda", "run", "-n", "ml-server", "python", "server.py" \
# # ]