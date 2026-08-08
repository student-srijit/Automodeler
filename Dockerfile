FROM public.ecr.aws/lambda/python:3.11

# Install build tools (needed by some packages) and dependencies
RUN yum install -y gcc gcc-c++ && yum clean all

COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir && \
    pip install -r requirements.txt --no-cache-dir

# Pre-download the model into the image so cold starts are fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
