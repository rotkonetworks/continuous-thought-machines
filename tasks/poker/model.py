"""
Poker CTM: Continuous Thought Machine for all poker situations.

Handles 2-10 players, any position, any stack depth.
Variable opponent count via cross-attention.

Architecture:
  hero_encoder(cards, pot, stack, position)
    → hero_embedding [d_model]
  opponent_encoder(stack, position, actions, stats) × N
    → opponent_embeddings [N, d_model]
  CTM recurrence:
    for each iteration:
      cross_attend(hero ↔ opponents)
      update propensities (transport + diffusion)
      synchronization → confidence measure
  propensity_head → action distribution

~10M params, fits on 8GB GPU with batch 256.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CardEncoder(nn.Module):
    """Encode cards as learned embeddings. 52 cards + padding."""
    def __init__(self, d_model: int):
        super().__init__()
        self.rank_embed = nn.Embedding(13, d_model // 2)
        self.suit_embed = nn.Embedding(4, d_model // 2)

    def forward(self, cards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        cards: [batch, num_cards] card indices 0-51 (or -1 for hidden)
        mask: [batch, num_cards] 1=visible, 0=hidden
        returns: [batch, num_cards, d_model]
        """
        # clamp to valid range for embedding lookup
        safe_cards = cards.clamp(0, 51)
        ranks = safe_cards % 13
        suits = safe_cards // 13
        r = self.rank_embed(ranks)  # [batch, n, d/2]
        s = self.suit_embed(suits)  # [batch, n, d/2]
        emb = torch.cat([r, s], dim=-1)  # [batch, n, d]
        return emb * mask.unsqueeze(-1)  # zero out hidden cards


class HeroEncoder(nn.Module):
    """Encode hero's full state: cards + game context."""
    def __init__(self, d_model: int):
        super().__init__()
        self.card_enc = CardEncoder(d_model)
        # 2 hole + 5 board = 7 cards → compress to d_model
        self.card_compress = nn.Sequential(
            nn.Linear(7 * d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        # game context: pot_norm, stack_norm, position, round, pot_odds, num_opponents
        self.context_enc = nn.Sequential(
            nn.Linear(8, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        # fuse cards + context
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, hole_cards, board_cards, board_mask, context):
        """
        hole_cards: [batch, 2]
        board_cards: [batch, 5]
        board_mask: [batch, 5] 1=visible
        context: [batch, 8] (pot, stack, opp_stack, position, round, pot_odds, to_call, num_opps)
        """
        hole_mask = torch.ones_like(hole_cards)
        all_cards = torch.cat([hole_cards, board_cards], dim=1)  # [batch, 7]
        all_mask = torch.cat([hole_mask, board_mask], dim=1)  # [batch, 7]

        card_emb = self.card_enc(all_cards, all_mask)  # [batch, 7, d]
        card_flat = card_emb.reshape(card_emb.shape[0], -1)  # [batch, 7*d]
        card_feat = self.card_compress(card_flat)  # [batch, d]

        ctx_feat = self.context_enc(context)  # [batch, d]
        hero = self.fuse(torch.cat([card_feat, ctx_feat], dim=-1))  # [batch, d]
        return hero


class OpponentEncoder(nn.Module):
    """Encode a single opponent's observable state. Shared across all opponents."""
    def __init__(self, d_model: int):
        super().__init__()
        # opponent features: stack_norm, position, last_action (one-hot 6),
        #                    vpip, pfr, aggression, hands_played
        self.net = nn.Sequential(
            nn.Linear(16, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, opp_features: torch.Tensor) -> torch.Tensor:
        """
        opp_features: [batch, max_opponents, 16]
        returns: [batch, max_opponents, d_model]
        """
        return self.net(opp_features)


class CrossAttention(nn.Module):
    """Hero attends to opponents. Who is the threat?"""
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hero: torch.Tensor, opponents: torch.Tensor, opp_mask: torch.Tensor) -> torch.Tensor:
        """
        hero: [batch, d_model]
        opponents: [batch, max_opps, d_model]
        opp_mask: [batch, max_opps] 1=active, 0=empty seat
        returns: [batch, d_model]
        """
        B, N, D = opponents.shape
        q = self.q_proj(hero).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(opponents).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(opponents).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # mask out empty seats
        mask = opp_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = attn.masked_fill(mask == 0, 0)  # zero out after softmax for empty seats

        out = (attn @ v).transpose(1, 2).reshape(B, D)
        return self.norm(hero + self.out_proj(out))


class PokerCTMCore(nn.Module):
    """
    CTM recurrence core for poker.

    Each iteration:
    1. Cross-attend hero ↔ opponents
    2. Update internal state via synapses
    3. Apply NLM (non-linear memory) processing
    4. Compute synchronization (confidence)
    """
    def __init__(self, d_model: int, memory_length: int = 64, synapse_depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.memory_length = memory_length

        self.cross_attn = CrossAttention(d_model, n_heads=4)

        # synapses: process (hero_state + memory) → new state
        layers = []
        for _ in range(synapse_depth):
            layers.extend([
                nn.Linear(d_model * 2 if len(layers) == 0 else d_model, d_model * 2),
                nn.GLU(),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            ])
        self.synapses = nn.Sequential(*layers)

        # NLM: per-neuron temporal processing
        self.nlm = nn.Sequential(
            nn.Linear(memory_length, memory_length * 2),
            nn.GELU(),
            nn.Linear(memory_length * 2, 1),
        )

        # initial state
        self.register_parameter(
            'start_trace',
            nn.Parameter(torch.zeros(d_model, memory_length).uniform_(
                -1/math.sqrt(d_model), 1/math.sqrt(d_model)
            ))
        )

    def init_state(self, batch_size: int, device: torch.device):
        """Initialize hidden state for a new hand."""
        trace = self.start_trace.unsqueeze(0).expand(batch_size, -1, -1).clone()
        return trace  # [batch, d_model, memory_length]

    def step(self, hero: torch.Tensor, opponents: torch.Tensor, opp_mask: torch.Tensor,
             state_trace: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One CTM iteration.
        Returns: (activated_state, updated_trace)
        """
        # 1. cross-attend to opponents
        hero_ctx = self.cross_attn(hero, opponents, opp_mask)

        # 2. combine with last activated state
        last_activated = self.nlm(state_trace).squeeze(-1)  # [batch, d_model]
        synapse_input = torch.cat([hero_ctx, last_activated], dim=-1)

        # 3. synapses → new state
        new_state = self.synapses(synapse_input)  # [batch, d_model]

        # 4. update trace (shift + append)
        state_trace = torch.cat([
            state_trace[:, :, 1:],
            new_state.unsqueeze(-1)
        ], dim=-1)

        # 5. activated state via NLM
        activated = self.nlm(state_trace).squeeze(-1)

        return activated, state_trace


class PokerCTM(nn.Module):
    """
    Full Poker CTM model.

    Input: game state (hero cards, board, pot, stacks, opponent info)
    Output: action distribution + value estimate
    """
    def __init__(
        self,
        d_model: int = 512,
        memory_length: int = 64,
        max_iterations: int = 32,
        num_actions: int = 8,  # fold, check, call, bet_0.33, bet_0.5, bet_0.75, bet_pot, allin
        synapse_depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_iterations = max_iterations
        self.num_actions = num_actions

        # encoders
        self.hero_enc = HeroEncoder(d_model)
        self.opp_enc = OpponentEncoder(d_model)

        # CTM core
        self.core = PokerCTMCore(d_model, memory_length, synapse_depth, dropout)

        # action head (policy)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_actions),
        )

        # value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # synchronization for adaptive iterations
        self.synch_proj = nn.Linear(d_model, 1)

    def forward(
        self,
        hole_cards: torch.Tensor,       # [batch, 2]
        board_cards: torch.Tensor,       # [batch, 5]
        board_mask: torch.Tensor,        # [batch, 5]
        context: torch.Tensor,           # [batch, 8]
        opp_features: torch.Tensor,      # [batch, max_opps, 16]
        opp_mask: torch.Tensor,          # [batch, max_opps]
        action_mask: torch.Tensor,       # [batch, num_actions] valid actions
        iterations: int | None = None,   # override thinking time
        hidden_state: torch.Tensor | None = None,  # persistent across hands
    ) -> dict:
        B = hole_cards.shape[0]
        device = hole_cards.device

        # encode
        hero = self.hero_enc(hole_cards, board_cards, board_mask, context)
        opponents = self.opp_enc(opp_features)

        # init or reuse hidden state
        if hidden_state is None:
            state_trace = self.core.init_state(B, device)
        else:
            state_trace = hidden_state

        n_iters = iterations or self.max_iterations

        # CTM recurrence
        activations = []
        for i in range(n_iters):
            activated, state_trace = self.core.step(hero, opponents, opp_mask, state_trace)
            activations.append(activated)

        # use final activation for policy + value
        final = activations[-1]

        # policy
        logits = self.policy_head(final)
        # mask invalid actions
        logits = logits.masked_fill(action_mask == 0, float('-inf'))
        action_probs = F.softmax(logits, dim=-1)

        # value
        value = self.value_head(final).squeeze(-1)

        # synchronization (confidence)
        synch = torch.sigmoid(self.synch_proj(final).squeeze(-1))

        return {
            'action_probs': action_probs,
            'value': value,
            'logits': logits,
            'confidence': synch,
            'hidden_state': state_trace,
            'activations': torch.stack(activations, dim=1),  # [B, iters, d_model]
        }


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # test model creation and forward pass
    model = PokerCTM(d_model=512, memory_length=64, max_iterations=16, num_actions=8)
    print(f'Parameters: {count_params(model):,}')

    B = 4
    out = model(
        hole_cards=torch.randint(0, 52, (B, 2)),
        board_cards=torch.randint(0, 52, (B, 5)),
        board_mask=torch.tensor([[1,1,1,0,0]]*B).float(),
        context=torch.randn(B, 8),
        opp_features=torch.randn(B, 9, 16),  # up to 9 opponents
        opp_mask=torch.tensor([[1,1,0,0,0,0,0,0,0]]*B).float(),  # 2 active opponents
        action_mask=torch.tensor([[1,0,1,1,1,1,1,1]]*B).float(),  # can't check
        iterations=16,
    )
    print(f'Action probs: {out["action_probs"].shape}')
    print(f'Value: {out["value"].shape}')
    print(f'Confidence: {out["confidence"].shape}')
    print(f'Activations: {out["activations"].shape}')
    print(f'Hidden state: {out["hidden_state"].shape}')
    print(f'Sample probs: {out["action_probs"][0].detach()}')
