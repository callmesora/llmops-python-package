# Use a multi-stage build to first get uv
FROM ghcr.io/astral-sh/uv:0.5.4 as uv
FROM python:3.12-slim

# Define build arguments
ARG GUARDRAILS_TOKEN

# Create a virtual environment with uv inside the container
RUN --mount=from=uv,source=/uv,target=./uv \
    ./uv venv /opt/venv

# We need to set this environment variable so that uv knows where
# the virtual environment is to install packages
ENV VIRTUAL_ENV=/opt/venv

# Make sure that the virtual environment is in the PATH so
# we can use the binaries of packages that we install such as pip
# without needing to activate the virtual environment explicitly
ENV PATH="/opt/venv/bin:$PATH"

# Copy the requirements file into the container
COPY requirements.txt .

# Install the packages with uv using --mount=type=cache to cache the downloaded packages
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=from=uv,source=/uv,target=./uv \
    ./uv pip install -r requirements.txt litserve==0.2.4

# Print to terminal Guardrail token to see if it is set correctly

# Set the GUARDRAILS_TOKEN as an environment variable
ENV GUARDRAILS_TOKEN=${GUARDRAILS_TOKEN}

RUN guardrails configure --enable-metrics --enable-remote-inferencing --token ${GUARDRAILS_TOKEN}

RUN guardrails hub install hub://tryolabs/restricttotopic


WORKDIR /app
COPY . /app

EXPOSE 8000
CMD ["python", "/app/server.py"]
