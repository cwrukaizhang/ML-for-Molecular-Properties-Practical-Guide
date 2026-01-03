ARG VERSION
FROM my-base-image:latest AS noenvpersist

ENV CONDA_SSL_VERIFY=false
ENV NB_PATH="/home/jovyan"
VOLUME ${NB_PATH}

ARG CONDA_CLEAN="conda clean -ay \
&& find /opt/conda/ -type f,l -name '*.a' -delete \
&& find /opt/conda/ -type f,l -name '*.pyc' -delete \
&& find /opt/conda/ -type f,l -name '*.js.map' -delete \
&& rm -rf /opt/conda/pkgs"

# Update notebook to 7.0 and install necessary packages
COPY environment_defaults.yml .
RUN conda env update -n base --file environment_defaults.yml && \
    conda install -c conda-forge -y cmake eigen rdkit && \
    pip install notebook==7.0.0 jupyter ipywidgets jupyter-server-proxy jupyter-contrib-nbextensions \
    jupyter_nbextensions_configurator nbresuse jupyterlab-git \
    cmaes numpy==1.26.4 pandas==2.0 scikit-learn==1.5.0 \
    lightgbm optuna ipython plotly matplotlib joblib tqdm ipycytoscape \
    rich openpyxl seaborn mordredcommunity --no-cache-dir && \
    #pip install --no-cache-dir --trusted-host download.pytorch.org \
    #torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html && \
    #pip install --no-cache-dir torch_geometric==2.3.1 captum && \
    #pip install --no-cache-dir --trusted-host data.pyg.org \
    #torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html && \
    sh -c "$CONDA_CLEAN"

# Enable extensions and clean up Jupyter lab build artifacts
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
    jupyter lab build --minimize=True && \
    jupyter lab clean

ENV SHELL=/bin/bash
ENV NB_PREFIX=/
CMD ["sh", "-c", "jupyter lab --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*'"]

FROM noenvpersist AS envpersist
ENV CONDA_ENVS_PATH="${NB_PATH}/envs"