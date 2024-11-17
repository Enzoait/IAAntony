# IMPORT
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# DEPENDENCES
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# INSTALL
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN pip install transformers datasets
RUN pip install sentencepiece
RUN pip install protobuf
RUN pip install unsloth
RUN pip install python-dotenv


WORKDIR /app
COPY . .

# EXE
CMD ["python3", "train_model.py"]
