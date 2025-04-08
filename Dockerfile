# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenCV dependencies (for cv2)
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . .

# Expose the port your app runs on
EXPOSE 5000

# Command to run your Flask app
CMD ["python", "together.py"]


