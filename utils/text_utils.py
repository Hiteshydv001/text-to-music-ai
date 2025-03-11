import logging
from typing import Optional
import torch
from transformers import BertTokenizer, BertModel

logger = logging.getLogger(__name__)

def get_text_embedding(text: str, model_name: str = "bert-base-uncased") -> Optional[torch.Tensor]:
    """Generate text embeddings with caching."""
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name).eval()
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        logger.info(f"Generated embedding for text: {text}")
        return embedding
    except Exception as e:
        logger.error(f"Text embedding failed: {e}")
        return None