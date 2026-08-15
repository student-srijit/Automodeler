FROM public.ecr.aws/lambda/python:3.11

# Install build tools
RUN yum install -y gcc gcc-c++ && yum clean all

# Install CPU-only PyTorch FIRST (avoids pulling 3.5GB of CUDA/cuDNN libs)
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir && \
    pip install -r requirements.txt --no-cache-dir

# Pre-bake the embedding model so cold starts are fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]

