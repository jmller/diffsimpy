# Use PyTorchs official GPU image as the base
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Create user with non-root privileges (otherwise I'll not be able to install modules in -editable mode)
RUN useradd -m appuser && echo "appuser:password" |chpasswd
RUN chsh -s /bin/bash appuser
USER appuser

# (Optional) Set the maintainer label
LABEL maintainer="johannes.mueller@mevis.fraunhofer.de"

# Set the working directory
USER root
RUN chsh -s /bin/bash
RUN echo "root:password" | chpasswd
WORKDIR /workspace 

# Install Git
RUN apt-get update && apt-get install -y git

# Copy your project files (e.g. Python scripts, Jupyter notebooks) into the image
RUN mkdir diffsimpy
COPY ./ /workspace/diffsimpy
RUN pip install -e ./diffsimpy 

COPY ./requirements.txt /
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

# Expose MLflow's default port
EXPOSE 5000

RUN chown appuser:appuser /workspace
RUN chmod -R 777 /workspace

USER appuser


