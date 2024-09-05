# Base Image - Using official Python image with version 3.9
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose port (optional, for debugging purposes or future API endpoints)
EXPOSE 8000

# Default command to run a bash shell (useful for manual running of scripts)
CMD ["bash"]
