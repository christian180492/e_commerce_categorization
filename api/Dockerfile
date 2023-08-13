# We use the official Python image as a base
FROM python:3.10

# We copy the code of the application in the container
COPY . /app

# Set the working directory
WORKDIR /app

# Install the application dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the port that Flask will use
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]

