# Use the official Python image for Python 3.10.9 as the base image
FROM python:3.10.9

# Install system dependencies including libGL.so.1
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

# Create and set the working directory in the container
WORKDIR /app

# Copy your Python application code into the container
COPY . /app

# Create a Python virtual environment and activate it
RUN python -m venv venv
RUN . venv/bin/activate

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any additional dependencies, if needed
# RUN pip install <additional_dependency>

# Install the FastAPI application dependencies
RUN pip install -r requirements.txt

# Copy the FastAPI application code into the container at /app
COPY main.py /app/

# Expose the port that the FastAPI application will run on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
