FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3-pip git && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

RUN pip install --no-cache-dir \
    "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    "trl>=0.19" "accelerate>=0.34" "peft>=0.11" "datasets>=2.19" \
    "transformers>=4.52" "huggingface_hub>=0.23" \
    fastapi uvicorn matplotlib

# Remove packages that register broken custom ops (string-annotated torch.Tensor sigs)
RUN pip uninstall -y bitsandbytes liger-kernel 2>/dev/null || true

COPY train_space.py /app/train_space.py

EXPOSE 7860

CMD ["python", "train_space.py"]
