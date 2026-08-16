FROM public.ecr.aws/lambda/python:3.11

# Set all cache and home directories to /tmp (writable at runtime) or /var/task (pre-baked into image)
ENV HOME=/tmp
ENV XDG_CACHE_HOME=/tmp
ENV HF_HOME=/var/task/hf_cache
ENV TORCH_HOME=/var/task/torch_cache

# Install build tools, C-dependencies for audio/image processing, and tar/gzip (needed for ccloud)
RUN yum install -y gcc gcc-c++ cmake libsndfile tar gzip zlib-devel libjpeg-devel && yum clean all

# Install CockroachDB ccloud CLI (CRITICAL for autonomous provisioning)
RUN curl -sL https://binaries.cockroachdb.com/ccloud/ccloud_linux-amd64_0.2.2.tar.gz | tar -xz && \
    mv ccloud /usr/local/bin/ccloud && \
    chmod +x /usr/local/bin/ccloud

# Install CPU-only PyTorch FIRST (avoids pulling 3.5GB of CUDA/cuDNN libs)
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install pre-compiled soxr wheel to bypass C++ compilation failure
RUN pip install --no-deps https://files.pythonhosted.org/packages/2b/97/cbce72f9c8b5c9c667eb55dc55be20a87c610dba55c0466c77498c1a8c97/soxr-0.3.7-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir && \
    pip install -r requirements.txt --no-cache-dir

# Install torchvision AFTER Pillow and Numpy are already installed via requirements.txt
RUN pip install --no-cache-dir torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

# Pre-bake the embedding model into /var/task/hf_cache so cold starts are instant!
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

# Copy function code
COPY lambda_function.py pipeline.py ${LAMBDA_TASK_ROOT}/
COPY core/ ${LAMBDA_TASK_ROOT}/core/

CMD ["lambda_function.lambda_handler"]
