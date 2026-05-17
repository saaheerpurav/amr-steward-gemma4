FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    "trl>=0.19" "accelerate>=0.34" "peft>=0.11" "datasets>=2.19" \
    "transformers>=4.52" "huggingface_hub>=0.23" \
    fastapi uvicorn

COPY train_space.py /app/train_space.py

EXPOSE 7860

CMD ["python", "train_space.py"]
