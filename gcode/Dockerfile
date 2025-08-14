# gcode Dockerfile
# Multi-stage build for production-ready container

# Build stage
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project context
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH=/root/.local/bin:$PATH

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create gcode user
RUN useradd --create-home --shell /bin/bash gcode

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/gcode/.local

# Copy the entire project from builder
COPY --from=builder /app .

# Install gcode globally
RUN pip install -e . && \
    chown -R gcode:gcode /app

# Switch to gcode user
USER gcode

# Set PATH for gcode user
ENV PATH=/home/gcode/.local/bin:$PATH

# Create .gcode directory for configuration
RUN mkdir -p /home/gcode/.gcode

# Expose port (if needed for web interface later)
EXPOSE 8000

# Default command
ENTRYPOINT ["gcode"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD gcode --help || exit 1
