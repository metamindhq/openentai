import os
from typing import List, Union
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import onnxruntime as ort
import openai

class EmbeddingGenerator:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.local_models = {}
        self.onnx_models = {}

    def generate_embedding(self, text: Union[str, List[str]], model_name: str) -> Union[np.ndarray, List[np.ndarray]]:
        if model_name.startswith("openai:"):
            return self._generate_openai_embedding(text, model_name[7:])
        elif model_name.startswith("huggingface:"):
            return self._generate_huggingface_embedding(text, model_name[11:])
        elif model_name.startswith("onnx:"):
            return self._generate_onnx_embedding(text, model_name[5:])
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def _generate_openai_embedding(self, text: Union[str, List[str]], model_name: str) -> Union[np.ndarray, List[np.ndarray]]:
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not set. Please set the OPENAI_API_KEY environment variable.")
        openai.api_key = self.openai_api_key
        response = openai.Embedding.create(input=text, model=model_name)
        if isinstance(text, str):
            return np.array(response['data'][0]['embedding'])
        return [np.array(item['embedding']) for item in response['data']]

    def _generate_huggingface_embedding(self, text: Union[str, List[str]], model_name: str) -> Union[np.ndarray, List[np.ndarray]]:
        if model_name not in self.local_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            self.local_models[model_name] = (tokenizer, model)
        else:
            tokenizer, model = self.local_models[model_name]

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings[0] if isinstance(text, str) else embeddings

    def _generate_onnx_embedding(self, text: Union[str, List[str]], model_name: str) -> Union[np.ndarray, List[np.ndarray]]:
        if model_name not in self.onnx_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            session = ort.InferenceSession(f"{model_name}.onnx")
            self.onnx_models[model_name] = (tokenizer, session)
        else:
            tokenizer, session = self.onnx_models[model_name]

        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
        outputs = session.run(None, dict(inputs))
        embeddings = outputs[0].mean(axis=1)
        return embeddings[0] if isinstance(text, str) else embeddings

    def get_embedding_size(self, model_name: str) -> int:
        sample_text = "This is a sample text to get the embedding size."
        embedding = self.generate_embedding(sample_text, model_name)
        return embedding.shape[0]

# Example usage:
# embedding_generator = EmbeddingGenerator()
# text = "Hello, world!"
# embedding = embedding_generator.generate_embedding(text, "openai:text-embedding-ada-002")
# embedding_size = embedding_generator.get_embedding_size("openai:text-embedding-ada-002")
# print(f"Embedding shape: {embedding.shape}, Size: {embedding_size}")
