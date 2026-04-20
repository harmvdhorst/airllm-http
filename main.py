import os
import logging
from model_manager import ModelManager

logging.basicConfig(level=logging.INFO)

# Load environment variables
AIRLLM_PRELOAD_MODELS = os.getenv('AIRLLM_PRELOAD_MODELS', '').split(',')
AIRLLM_MODEL = os.getenv('AIRLLM_MODEL')
AIRLLM_MAX_MODELS = int(os.getenv('AIRLLM_MAX_MODELS', 5))

model_manager = ModelManager(max_models=AIRLLM_MAX_MODELS)

def preload_models():
    if AIRLLM_PRELOAD_MODELS:
        for model_name in AIRLLM_PRELOAD_MODELS:
            model_manager.load_model(model_name.strip())
        logging.info(f'Preloaded models: {AIRLLM_PRELOAD_MODELS}')

if __name__ == '__main__':
    preload_models()
    # Your code for handling requests goes here.
