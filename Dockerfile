# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on (change if needed)
EXPOSE 5000

# Command to run the app (change 'app.py' and 'app' if different)
CMD ["python", "app/app.py"]
