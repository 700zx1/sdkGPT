FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for SDK docs
RUN mkdir -p /app/sdk_docs

# Set default command
CMD ["python", "main.py"]
