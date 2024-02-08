# The builder image, used to build the virtual environment
FROM python:3.11-buster
# Define system variables
## Define poetry version
ENV POETRY_VERSION=1.4.0
# Set the environment variable to enable non-interactive mode
ENV POETRY_NO_INTERACTION=1
# Poetry creates the virtual environment within the project's directory
ENV POETRY_VIRTUALENVS_IN_PROJECT=1
# Poetry will create a new virtual environment, even if one is already present
ENV POETRY_VIRTUALENVS_CREATE=1
# Poetry will use the specified directory as the cache location
ENV POETRY_CACHE_DIR=/tmp/poetry_cache
# Change Working Directory to /app
WORKDIR /app
# Copy poetry packages required to run/install
COPY pyproject.toml poetry.lock README.md ./
# Install packages
## Install poetry
### Install specified poetry package
## Install dependencies
### Instruct Buildkit to mount and manage a folder for caching reasons
### Avoid installing development dependencies by using --without dev
### Remove the cache once installed by using rm -rf $POETRY_CACHE_DIR
### Install dependencies before copying code by using --no-root
RUN pip install "poetry==$POETRY_VERSION"
#RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --without dev --no-root 
RUN poetry install --without dev --no-root 
COPY quantified_strategies ./quantified_strategies
RUN poetry install --without dev 
ENTRYPOINT ["poetry", "run", "python", "-m", "quantified_strategies.main"]
# Build: docker build -t "dockername" .
# Run: docker run -p 9999:9999 "dockername"
