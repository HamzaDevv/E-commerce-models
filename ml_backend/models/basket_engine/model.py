"""
BasketGPT — A lightweight causal transformer for basket completion.

Architecture:
  - Product embedding (vocab_size → embed_dim)
  - RoPE (Rotary Position Embeddings) — no learned position table
  - N × Transformer decoder blocks with causal self-attention
  - Tied output projection (shares weights with embedding)

Trained with next-token prediction on order sequences.
At inference: autoregressive generation of basket completions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from pathlib import Path


# ──────────────────────────────────────────────
# Rotary Position Embeddings (RoPE)
# ──────────────────────────────────────────────

def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute the complex exponential frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)  # (max_seq_len, dim//2)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Apply rotary embeddings to query/key tensors.
    x: (batch, n_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim//2) — will be broadcast
    """
    seq_len = x.shape[2]
    head_dim = x.shape[3]
    
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    
    # Split x into two halves for rotation
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    
    # Apply rotation
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], dim=-1)
    
    return rotated


# ──────────────────────────────────────────────
# Multi-Head Self-Attention with RoPE
# ──────────────────────────────────────────────

class RoPEAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embeddings and causal mask."""
    
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1, max_seq_len: int = 50):
        super().__init__()
        assert embed_dim % n_heads == 0
        
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.embed_dim = embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Precompute RoPE frequencies
        cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)
        
        # Causal mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)
    
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to queries and keys
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        
        # Scaled dot-product attention with causal mask
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = attn.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


# ──────────────────────────────────────────────
# Transformer Block
# ──────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual"""
    
    def __init__(self, embed_dim: int, n_heads: int, ffn_dim: int, dropout: float = 0.1, max_seq_len: int = 50):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = RoPEAttention(embed_dim, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ──────────────────────────────────────────────
# BasketGPT Model
# ──────────────────────────────────────────────

class BasketGPT(nn.Module):
    """
    A mini-GPT for shopping basket completion.
    
    Treats product IDs as tokens. Trained with next-token prediction
    on order sequences. Uses RoPE for position information.
    Output projection is weight-tied with the input embedding.
    
    Dataset-specific defaults:
      - vocab_size=49,691 (49,688 products + PAD:0 + BOS:49689 + UNK:49690)
      - max_seq_len=50 (covers 99th percentile basket size of 35 + room)
      - embed_dim=64, n_heads=4, n_layers=2, ffn_dim=256
    """
    
    # Special token IDs
    PAD_TOKEN = 0
    BOS_TOKEN = 49689
    UNK_TOKEN = 49690
    
    def __init__(
        self,
        vocab_size: int = 49691,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 256,
        max_seq_len: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout
        
        # Token embedding (no position embedding — RoPE handles it)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.PAD_TOKEN)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ffn_dim, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # Output head — weight-tied with token embedding
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight  # Weight tying
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len) — product ID tokens
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        x = self.token_embedding(input_ids)
        x = self.embed_dropout(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_final(x)
        logits = self.output_head(x)
        
        return logits
    
    def get_config(self) -> dict:
        """Return model configuration as a dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'ffn_dim': self.ffn_dim,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout_rate,
            'pad_token': self.PAD_TOKEN,
            'bos_token': self.BOS_TOKEN,
            'unk_token': self.UNK_TOKEN,
        }
    
    @classmethod
    def from_config(cls, config: dict) -> 'BasketGPT':
        """Instantiate model from a config dict."""
        return cls(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            ffn_dim=config['ffn_dim'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout'],
        )
    
    @torch.inference_mode()
    def generate(
        self,
        cart_ids: list[int],
        num_suggestions: int = 5,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> list[dict]:
        """
        Autoregressive basket completion generation.
        
        Like GPT text generation: predict one product, append it,
        predict the next, and so on.
        
        Args:
            cart_ids: List of product IDs currently in the cart
            num_suggestions: How many products to generate
            temperature: Sampling temperature (lower = more conservative)
            top_k: Top-K sampling (only sample from top K most likely products)
            
        Returns:
            List of dicts: [{'product_id': int, 'confidence': float, 'rank': int}, ...]
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Build initial sequence: [BOS, p1, p2, ..., pN]
        sequence = [self.BOS_TOKEN] + cart_ids
        
        # Clamp product IDs to valid range
        sequence = [
            pid if 0 < pid < self.BOS_TOKEN else self.UNK_TOKEN 
            for pid in sequence
        ]
        sequence[0] = self.BOS_TOKEN  # Restore BOS
        
        # Truncate if too long (leave room for generation)
        max_input = self.max_seq_len - num_suggestions
        if len(sequence) > max_input:
            sequence = [self.BOS_TOKEN] + sequence[-(max_input - 1):]
        
        already_in_cart = set(cart_ids)
        suggestions = []
        
        for rank in range(1, num_suggestions + 1):
            input_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
            logits = self(input_tensor)
            
            # Get logits at the last position
            next_logits = logits[0, -1, :].float()
            
            # Apply temperature
            next_logits = next_logits / temperature
            
            # Mask special tokens and already-in-cart items
            next_logits[self.PAD_TOKEN] = float('-inf')
            next_logits[self.BOS_TOKEN] = float('-inf')
            next_logits[self.UNK_TOKEN] = float('-inf')
            for pid in already_in_cart:
                if 0 <= pid < self.vocab_size:
                    next_logits[pid] = float('-inf')
            for s in suggestions:
                next_logits[s['product_id']] = float('-inf')
            
            # Top-K filtering
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(next_logits, min(top_k, self.vocab_size))
                mask = torch.full_like(next_logits, float('-inf'))
                mask.scatter_(0, topk_idx, topk_vals)
                next_logits = mask
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            confidence = probs[next_token].item()
            
            suggestions.append({
                'product_id': next_token,
                'confidence': round(confidence, 6),
                'rank': rank,
            })
            
            # Autoregressive: append prediction to sequence
            sequence.append(next_token)
        
        return suggestions
    
    def count_parameters(self) -> dict:
        """Count model parameters by component."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embedding = self.token_embedding.weight.numel()
        transformer = sum(
            p.numel() for block in self.blocks for p in block.parameters()
        )
        return {
            'total': total,
            'trainable': trainable,
            'embedding': embedding,
            'transformer_blocks': transformer,
            'final_ln': sum(p.numel() for p in self.ln_final.parameters()),
            'note': 'Output head is weight-tied with embedding (0 extra params)',
        }
