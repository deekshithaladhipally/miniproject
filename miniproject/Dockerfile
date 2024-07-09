# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /miniproject

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /miniproject/requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir -r /miniproject/requirements.txt

# Copy the rest of the application code into the container at /app
COPY . /miniproject

# Expose the port on which the Flask app will run
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask application
CMD flask run
