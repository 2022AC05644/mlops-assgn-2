# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code and the model file
COPY . .

# Expose the port your app runs on
EXPOSE 5000

# Define the command to run your app
CMD ["python", "app.py"]