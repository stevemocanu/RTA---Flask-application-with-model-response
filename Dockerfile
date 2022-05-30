# This file defines the Docker container that will contain the Flask app.

# From the source image
FROM python:3.8.5

# Identify maintainer
LABEL maintainer = "sm110054@student.sgh.waw.pl"

# Set the default working directory
WORKDIR /app/

# Copy requirements.txt outside the container
# to /app/ inside the container
COPY requirements.txt /app/

# Install required packages
RUN pip install -r ./requirements.txt

# Copy app.py, train.py, __init__.py outside the container
# to /app/ inside the container
COPY __init__.py train.py app.py /app/

# Copy model.pkl outside the container
# to /app/ inside the container
COPY model.pkl /app/

# Expose the container's port 3333
EXPOSE 3333

# When the container starts, run this
ENTRYPOINT python ./app.py