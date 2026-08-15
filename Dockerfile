FROM public.ecr.aws/lambda/python:3.11

# Set all cache and home directories to /tmp (writable at runtime) or /var/task (pre-baked into image)
ENV HOME=/tmp
ENV XDG_CACHE_HOME=/tmp
ENV HF_HOME=/var/task/hf_cache
ENV TORCH_HOME=/var/task/torch_cache

# Install build tools and tar/gzip (needed for ccloud)
RUN yum install -y gcc gcc-c++ tar gzip && yum clean all

# Install CockroachDB ccloud CLI (CRITICAL for autonomous provisioning)
RUN curl -sL https://binaries.cockroachdb.com/ccloud/ccloud_linux-amd64_0.3.0.tar.gz | tar -xz && \
    mv ccloud /usr/local/bin/ccloud && \
    chmod +x /usr/local/bin/ccloud

# Install CPU-only PyTorch FIRST (avoids pulling 3.5GB of CUDA/cuDNN libs)
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir && \
    pip install -r requirements.txt --no-cache-dir

# Pre-bake the embedding model into /var/task/hf_cache so cold starts are instant!
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
