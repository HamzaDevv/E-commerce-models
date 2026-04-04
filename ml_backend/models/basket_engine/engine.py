"""
BasketCompletionEngine — Runtime inference engine for BasketGPT.

Loads pre-trained model weights and product metadata from disk.
Provides basket completion recommendations via autoregressive generation.
"""

import os
import json
import pickle
import torch
from models.basket_engine.model import BasketGPT


class BasketCompletionEngine:
    """
    Inference engine for basket completion using a trained BasketGPT model.
    
    Usage:
        engine = BasketCompletionEngine()
        results = engine.recommend([24852, 13176, 21137], num_suggestions=5)
    """
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.data_dir = os.path.join(base_dir, 'data', 'basket')
        else:
            self.data_dir = data_dir
        
        self._load_model()
    
    def _load_model(self):
        """Load model weights, config, and product lookup from disk."""
        print("Loading BasketGPT model...")
        
        config_path = os.path.join(self.data_dir, 'basket_gpt_config.json')
        weights_path = os.path.join(self.data_dir, 'basket_gpt.pt')
        lookup_path = os.path.join(self.data_dir, 'product_lookup.pkl')
        
        # Check if all files exist
        for path, name in [(config_path, 'config'), (weights_path, 'weights'), (lookup_path, 'product_lookup')]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"BasketGPT {name} not found at {path}. "
                    "Please train the model first using the Colab notebook."
                )
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Instantiate model from config
        self.model = BasketGPT.from_config(self.config)
        
        # Load trained weights
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Load product lookup
        with open(lookup_path, 'rb') as f:
            self.product_lookup = pickle.load(f)
        
        param_info = self.model.count_parameters()
        print(f"BasketGPT loaded: {param_info['total']:,} params | "
              f"Embed: {param_info['embedding']:,} | "
              f"Transformer: {param_info['transformer_blocks']:,}")
        print(f"Config: {self.config['n_layers']}L / {self.config['n_heads']}H / "
              f"{self.config['embed_dim']}D / vocab={self.config['vocab_size']}")
    
    def get_product_name(self, product_id: int) -> str:
        """Look up a product name by ID."""
        return self.product_lookup.get(product_id, f"Unknown Product (ID: {product_id})")
    
    def recommend(
        self,
        cart_product_ids: list[int],
        num_suggestions: int = 5,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> dict:
        """
        Generate basket completion recommendations.
        
        Args:
            cart_product_ids: Product IDs currently in the user's cart
            num_suggestions: Number of products to suggest (1-10)
            temperature: Sampling temperature (0.1-2.0). 
                         Lower = more conservative/popular items.
                         Higher = more diverse/surprising suggestions.
            top_k: Only sample from top-K most likely products
            
        Returns:
            Dict with cart info, suggestions, and model metadata
        """
        # Validate inputs
        num_suggestions = max(1, min(num_suggestions, 10))
        temperature = max(0.1, min(temperature, 2.0))
        top_k = max(1, min(top_k, 200))
        
        if not cart_product_ids:
            return {
                "cart_products": [],
                "suggestions": [],
                "model": "BasketGPT",
                "generation_mode": "autoregressive",
                "error": "Cart is empty — add products to get suggestions."
            }
        
        # Generate suggestions
        raw_suggestions = self.model.generate(
            cart_ids=cart_product_ids,
            num_suggestions=num_suggestions,
            temperature=temperature,
            top_k=top_k,
        )
        
        # Enrich with product names
        cart_products = [
            {
                "product_id": pid,
                "product_name": self.get_product_name(pid)
            }
            for pid in cart_product_ids
        ]
        
        suggestions = []
        for s in raw_suggestions:
            suggestions.append({
                "product_id": s['product_id'],
                "product_name": self.get_product_name(s['product_id']),
                "confidence": s['confidence'],
                "rank": s['rank'],
            })
        
        return {
            "cart_products": cart_products,
            "suggestions": suggestions,
            "model": "BasketGPT",
            "generation_mode": "autoregressive",
            "params": {
                "temperature": temperature,
                "top_k": top_k,
                "num_suggestions": num_suggestions,
            }
        }
