FROM public.ecr.aws/lambda/python:3.11

# Install build tools (needed by some packages) and dependencies
RUN yum install -y gcc gcc-c++ postgresql-devel tar gzip || yum install -y gcc gcc-c++ libpq-devel tar gzip && yum clean all

# Install CockroachDB ccloud CLI
RUN curl -sL https://binaries.cockroachdb.com/ccloud/ccloud_linux-amd64_0.3.0.tar.gz | tar -xz && \
    mv ccloud /usr/local/bin/ccloud && \
    chmod +x /usr/local/bin/ccloud

COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir && \
    pip install -r requirements.txt --no-cache-dir

# Pre-download the model into the image so cold starts are fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
