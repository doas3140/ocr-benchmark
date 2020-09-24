FROM continuumio/miniconda3:latest
WORKDIR /code
# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]
# fixes opencv error: https://github.com/ContinuumIO/docker-images/issues/49
RUN apt-get update && apt-get install libgl1-mesa-glx -y
COPY . .
RUN conda env create -f conda_environment.yml
RUN echo "conda activate ml-server" > ~/.bashrc
RUN conda init bash
RUN conda activate ml-server
EXPOSE 5000

ENTRYPOINT ["conda", "run", "-n", "ml-server", "python", "server.py"]