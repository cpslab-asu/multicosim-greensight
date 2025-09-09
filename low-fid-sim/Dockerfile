# Dockerfile for Optical Flow Attack Low-Fidelity Simulation
# Based on current conda py310 environment dependencies

FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Update conda and install basic dependencies
RUN conda update -n base -c defaults conda && \
    conda install -c conda-forge python=3.10.18

# Install conda-forge packages that are available there
RUN conda install -c conda-forge \
    pyfmi=2.9.7 \
    numpy=1.26.4 \
    assimulo=3.2.9 \
    attrs=23.2.0 \
    cython=3.1.3

# Install pip packages
RUN pip install --no-cache-dir \
    antlr4-python3-runtime==4.7 \
    matplotlib==3.10.6 \
    pandas==2.3.2 \
    scipy==1.15.3 \
    psy-taliro==1.0.0 \
    rtamt==0.3.5 \
    typing-extensions \
    docker==7.1.0 \
    dill==0.4.0 \
    contourpy==1.3.2 \
    cycler==0.12.1 \
    choreographer==1.0.10 \
    certifi \
    charset-normalizer==3.4.3 \
    click>=8.2.1 \
    casadi>=3.7.1 \
    openpyxl>=3.1.5 \
    nanoid \
    pyzmq

# Install multicosim from the specific git revision used in multicosim-greensight
RUN pip install git+https://github.com/cpslab-asu/multicosim.git@0198fc0f573365b7ec7838b3a0261e20ae4323f5

# Copy the low-fid-sim directory contents
COPY . /app/

# Set environment variables
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Create directories for outputs if they don't exist
RUN mkdir -p /app/figures /app/results

# Make the script executable
RUN chmod +x optflowattackdrone.py

# Expose any ports if needed (optional)
# EXPOSE 8080

# Set the default command to run the optical flow attack
CMD ["python", "optflowattackdrone.py"]
