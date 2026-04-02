# ============================================================
# Stage 1: Build the React frontend
# ============================================================
FROM node:18-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ============================================================
# Stage 2: Python backend + serve built frontend
# ============================================================
FROM python:3.11-slim

# HuggingFace Spaces expects port 7860
ENV PORT=7860

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY main.py ./
COPY scripts/ ./scripts/

# Copy model weights
COPY models/ ./models/

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
