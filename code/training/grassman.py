
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F



@dataclass
class GrassmannConfig:
    vocab_size: int = 50_257          # GPT-2 tokenizer
    max_seq_len: int = 1_024
    d_model: int = 640
    n_layers: int = 16
    d_ff: int = 2_560                 # 4 * d_model
    r: int = 24                       # reduced grassmann dimension; C(24,2)=276
    offsets: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    d_geom: int = 256                 
    dropout: float = 0.1
    eps: float = 1e-6
    tie_weights: bool = True


@dataclass
class TransformerConfig:
    vocab_size: int = 50_257
    max_seq_len: int = 1_024
    d_model: int = 640
    n_layers: int = 16
    n_heads: int = 10                 # 64 per head
    d_ff: int = 2_560
    dropout: float = 0.1
    tie_weights: bool = True


# modules shared by both models

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class _BaseLM(nn.Module):
    """Shared embedding -> blocks -> norm -> head logic for both model variants."""
    cfg: GrassmannConfig | TransformerConfig
    token_emb: nn.Embedding
    pos_emb: nn.Embedding
    drop: nn.Dropout
    blocks: nn.ModuleList
    final_norm: nn.LayerNorm
    lm_head: nn.Linear

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _embed_tokens(
        self,
        input_ids: torch.Tensor,
        *,
        position_offset: int = 0,
    ) -> torch.Tensor:
        B, L = input_ids.shape
        del B
        assert 0 <= position_offset <= self.cfg.max_seq_len
        assert position_offset + L <= self.cfg.max_seq_len
        pos = torch.arange(
            position_offset,
            position_offset + L,
            device=input_ids.device,
        ).unsqueeze(0)
        return self.drop(self.token_emb(input_ids) + self.pos_emb(pos))

    def supports_stateful_decode(self) -> bool: # helper for later
        return False

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, L = input_ids.shape
        assert L <= self.cfg.max_seq_len

        x = self._embed_tokens(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
        return out

    def num_params(self) -> int:
        """Count unique parameters (shared weights counted once)."""
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total


# my attempt at a causal grassmann mixer


class CausalGrassmannMixer(nn.Module):

    def __init__(
        self,
        d_model: int,
        r: int,
        offsets: tuple[int, ...],
        dropout: float,
        eps: float,
        d_geom: int,
    ) -> None:
        super().__init__()
        self.r = r
        self.offsets = tuple(sorted({int(o) for o in offsets if o > 0}))
        self.max_offset = max(self.offsets, default=0)
        self.eps = eps

        self.red = nn.Linear(d_model, r)

        plu_dim = r * (r - 1) // 2
        self.geom_proj = nn.Sequential(
            nn.Linear(plu_dim, d_geom),
            nn.GELU(),
            nn.Linear(d_geom, d_model),
        )

        self.gate = nn.Linear(2 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

        idx_i, idx_j = torch.triu_indices(r, r, offset=1)
        self.register_buffer("idx_i", idx_i, persistent=False)
        self.register_buffer("idx_j", idx_j, persistent=False)

    def _plucker(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        p = u[..., self.idx_i] * v[..., self.idx_j] - u[..., self.idx_j] * v[..., self.idx_i]
        norm = p.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return p / norm

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h : (B, L, d_model)
        returns (B, L, d_model)
        """
        B, L, D = h.shape
        z = self.red(h) 

        geom_sum = h.new_zeros(B, L, D)
        count = h.new_zeros(B, L, 1)

        for delta in self.offsets:
            if delta >= L:
                continue

            pad = z.new_zeros(B, delta, self.r)
            z_past = torch.cat([pad, z[:, :-delta]], dim=1)  

            p = self._plucker(z_past, z)
            g_delta = self.geom_proj(p)  

            mask = z.new_zeros(B, L, 1)
            mask[:, delta:] = 1.0

            geom_sum = geom_sum + g_delta * mask
            count = count + mask

        g = geom_sum / count.clamp_min(1.0)

        alpha = torch.sigmoid(self.gate(torch.cat([h, g], dim=-1)))
        out = alpha * h + (1.0 - alpha) * g
        return self.drop(out)

    def decode_step(
        self,
        h_t: torch.Tensor,
        state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        recent_z = None if state is None else state.get("recent_z")
        if recent_z is None:
            recent_z = h_t.new_empty(h_t.size(0), 0, self.r)

        z_t = self.red(h_t)  
        geom_terms: list[torch.Tensor] = []
        for delta in self.offsets:
            if delta > recent_z.size(1):
                continue
            z_past = recent_z[:, -delta, :]
            p = self._plucker(z_past, z_t)
            geom_terms.append(self.geom_proj(p))

        if geom_terms:
            g_t = torch.stack(geom_terms, dim=0).mean(dim=0)
        else:
            g_t = torch.zeros_like(h_t)

        alpha = torch.sigmoid(self.gate(torch.cat([h_t, g_t], dim=-1)))
        out = self.drop(alpha * h_t + (1.0 - alpha) * g_t)

        updated_recent_z = torch.cat([recent_z, z_t.unsqueeze(1)], dim=1)
        if self.max_offset > 0 and updated_recent_z.size(1) > self.max_offset:
            updated_recent_z = updated_recent_z[:, -self.max_offset :, :]
        return out, {"recent_z": updated_recent_z}


class GrassmannBlock(nn.Module):
    def __init__(self, cfg: GrassmannConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.mix = CausalGrassmannMixer( # my attempt at a causal grassmann mixer
            d_model=cfg.d_model,
            r=cfg.r,
            offsets=cfg.offsets,
            dropout=cfg.dropout,
            eps=cfg.eps,
            d_geom=cfg.d_geom,
        )
        
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout) # standard perceptron feedforward network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

    def decode_step(
        self,
        x_t: torch.Tensor,
        state: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        h_t = self.norm1(x_t)
        mix_state = None if state is None else state.get("mix")
        mix_out, next_mix_state = self.mix.decode_step(h_t, mix_state)
        x_t = x_t + mix_out
        x_t = x_t + self.ff(self.norm2(x_t))
        return x_t, {"mix": next_mix_state}


class GrassmannLM(_BaseLM):
    """full causal grassmann language model."""

    MODEL_TYPE = "grassmann"

    def __init__(self, cfg: GrassmannConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [GrassmannBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def supports_stateful_decode(self) -> bool: # helper for later
        return True

    def init_decode_state(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> dict[str, object]:
        return {
            "next_position": 0,
            "blocks": [None for _ in self.blocks],
            "batch_size": batch_size,
            "device": str(device),
        }

    def _decode_hidden_step(
        self,
        x_t: torch.Tensor,
        state: dict[str, object],
    ) -> tuple[torch.Tensor, dict[str, object]]:
        block_states = state["blocks"]
        next_block_states: list[dict[str, dict[str, torch.Tensor]]] = []
        for block, block_state in zip(self.blocks, block_states, strict=True):
            x_t, next_block_state = block.decode_step(x_t, block_state)
            next_block_states.append(next_block_state)
        next_state = {
            "next_position": int(state["next_position"]) + 1,
            "blocks": next_block_states,
            "batch_size": state["batch_size"],
            "device": state["device"],
        }
        return x_t, next_state

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor | dict[str, object]]:
        B, L = input_ids.shape
        assert L <= self.cfg.max_seq_len

        x = self._embed_tokens(input_ids)
        state = self.init_decode_state(batch_size=B, device=input_ids.device)
        logits_steps: list[torch.Tensor] = []

        for t in range(L):
            x_t, state = self._decode_hidden_step(x[:, t, :], state)
            logits_steps.append(self.lm_head(self.final_norm(x_t)))

        if logits_steps:
            logits = torch.stack(logits_steps, dim=1)
        else:
            logits = x.new_empty(B, 0, self.cfg.vocab_size)
        return {"logits": logits, "state": state}

    @torch.no_grad()
    def decode_step(
        self,
        input_ids: torch.Tensor,
        state: dict[str, object],
    ) -> dict[str, torch.Tensor | dict[str, object]]:
        B, L = input_ids.shape
        assert L == 1
        assert B == int(state["batch_size"])
        position = int(state["next_position"])
        assert position < self.cfg.max_seq_len

        x_t = self._embed_tokens(input_ids, position_offset=position)[:, 0, :]
        x_t, next_state = self._decode_hidden_step(x_t, state)
        logits = self.lm_head(self.final_norm(x_t)).unsqueeze(1)
        return {"logits": logits, "state": next_state}

# transformer model

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.max_seq_len = cfg.max_seq_len
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        causal_mask = torch.triu(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 3:
            B, L, _ = t.shape
            return t.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        if t.dim() == 2:
            B, _ = t.shape
            return t.view(B, self.n_heads, self.d_head).unsqueeze(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.causal_mask[:L, :L], float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))

        y = (att @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.drop(self.out_proj(y))

    def prefill(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.causal_mask[:L, :L], float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))

        y = (att @ v).transpose(1, 2).contiguous().view(B, L, D)
        out = self.drop(self.out_proj(y))
        return out, {"k": k, "v": v}

    def decode_step(
        self,
        x_t: torch.Tensor,
        state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, D = x_t.shape
        q_t, k_t, v_t = self.qkv(x_t).chunk(3, dim=-1)
        q_t = self._split_heads(q_t)  # (B, H, 1, d_head)
        k_t = self._split_heads(k_t)
        v_t = self._split_heads(v_t)

        if state is None:
            cached_k = x_t.new_empty(B, self.n_heads, 0, self.d_head)
            cached_v = x_t.new_empty(B, self.n_heads, 0, self.d_head)
        else:
            cached_k = state["k"]
            cached_v = state["v"]

        k = torch.cat([cached_k, k_t], dim=2)
        v = torch.cat([cached_v, v_t], dim=2)
        if k.size(2) > self.max_seq_len:
            k = k[:, :, -self.max_seq_len :, :]
            v = v[:, :, -self.max_seq_len :, :]

        att = (q_t @ k.transpose(-2, -1)) * self.scale
        att = self.drop(F.softmax(att, dim=-1))

        y = (att @ v).transpose(1, 2).contiguous().view(B, D)
        out = self.drop(self.out_proj(y))
        return out, {"k": k, "v": v}


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

    def prefill(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        attn_in = self.norm1(x)
        attn_out, attn_state = self.attn.prefill(attn_in)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, {"attn": attn_state}

    def decode_step(
        self,
        x_t: torch.Tensor,
        state: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        attn_state = None if state is None else state.get("attn")
        attn_out, next_attn_state = self.attn.decode_step(self.norm1(x_t), attn_state)
        x_t = x_t + attn_out
        x_t = x_t + self.ff(self.norm2(x_t))
        return x_t, {"attn": next_attn_state}


class TransformerLM(_BaseLM):
    """standard transformer model for comparison with grassmann architecture"""

    MODEL_TYPE = "transformer"

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def supports_stateful_decode(self) -> bool: # helper for later
        return True

    def init_decode_state(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> dict[str, object]:
        return {
            "next_position": 0,
            "blocks": [None for _ in self.blocks],
            "batch_size": batch_size,
            "device": str(device),
        }

    def _decode_hidden_step(
        self,
        x_t: torch.Tensor,
        state: dict[str, object],
    ) -> tuple[torch.Tensor, dict[str, object]]:
        block_states = state["blocks"]
        next_block_states: list[dict[str, dict[str, torch.Tensor]]] = []
        for block, block_state in zip(self.blocks, block_states, strict=True):
            x_t, next_block_state = block.decode_step(x_t, block_state)
            next_block_states.append(next_block_state)
        next_state = {
            "next_position": int(state["next_position"]) + 1,
            "blocks": next_block_states,
            "batch_size": state["batch_size"],
            "device": state["device"],
        }
        return x_t, next_state

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor | dict[str, object]]:
        B, L = input_ids.shape
        assert L <= self.cfg.max_seq_len

        x = self._embed_tokens(input_ids)
        state = self.init_decode_state(batch_size=B, device=input_ids.device)
        next_block_states: list[dict[str, dict[str, torch.Tensor]]] = []
        for block in self.blocks:
            x, block_state = block.prefill(x)
            next_block_states.append(block_state)
        state["blocks"] = next_block_states
        state["next_position"] = L

        logits = self.lm_head(self.final_norm(x))
        return {"logits": logits, "state": state}

    @torch.no_grad()
    def decode_step(
        self,
        input_ids: torch.Tensor,
        state: dict[str, object],
    ) -> dict[str, torch.Tensor | dict[str, object]]:
        B, L = input_ids.shape
        assert L == 1
        assert B == int(state["batch_size"])
        position = int(state["next_position"])
        assert position < self.cfg.max_seq_len

        x_t = self._embed_tokens(input_ids, position_offset=position)[:, 0, :]
        x_t, next_state = self._decode_hidden_step(x_t, state)
        logits = self.lm_head(self.final_norm(x_t)).unsqueeze(1)
        return {"logits": logits, "state": next_state}
