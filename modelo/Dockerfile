# We use a python base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# We copy the code of the application in the container
COPY . /app/

# Install the application dependencies
RUN pip install -r requirements.txt

# We execute the file modelo.py
CMD ["python", "modelo.py"]
