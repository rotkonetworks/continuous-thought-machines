"""CTM decoder for quantum error correction.

Follows the SORT model pattern: no backbone, no attention,
raw syndrome vector as input. The CTM "thinks" over multiple
ticks to decode the syndrome into a logical error class.
"""

import torch
import numpy as np
from models.ctm import ContinuousThoughtMachine


class ContinuousThoughtMachineQEC(ContinuousThoughtMachine):
    """CTM adapted for QEC syndrome decoding.

    Input: flattened syndrome history [B, R * n_stabilizers]
    Output: logical error class predictions [B, 4, T]
        class 0 = I (no logical error)
        class 1 = X logical error
        class 2 = Z logical error
        class 3 = Y logical error (both X and Z)
    """

    def __init__(self, syndrome_dim, d_model=256, iterations=32,
                 n_synch_out=64, synapse_depth=3, memory_length=16,
                 memory_hidden_dims=16, neuron_select_type='random-pairing',
                 n_random_pairing_self=0, dropout=0):
        super().__init__(
            iterations=iterations,
            d_model=d_model,
            d_input=syndrome_dim,  # will be overridden by lazy init
            heads=0,               # no attention
            n_synch_out=n_synch_out,
            n_synch_action=0,      # no action sync
            synapse_depth=synapse_depth,
            memory_length=memory_length,
            deep_nlms=True,
            memory_hidden_dims=memory_hidden_dims,
            do_layernorm_nlm=False,
            backbone_type='none',
            positional_embedding_type='none',
            out_dims=4,            # I, X, Z, Y
            prediction_reshaper=[-1],
            dropout=dropout,
            neuron_select_type=neuron_select_type,
            n_random_pairing_self=n_random_pairing_self,
        )

        # Nullify attention (same as SORT)
        self.neuron_select_type_action = None
        self.synch_representation_size_action = None
        self.attention = None
        self.q_proj = None
        self.kv_proj = None

    def forward(self, x, track=False):
        """Forward pass for QEC decoding.

        Args:
            x: [B, syndrome_dim] flattened syndrome history
            track: if True, return extended tracking info

        Returns:
            predictions: [B, 4, T] logits per tick
            certainties: [B, 2, T]
            sync_out: [B, n_synch_out]
        """
        B = x.size(0)
        device = x.device

        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []

        # Initialize recurrent state
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)

        predictions = torch.empty(B, self.out_dims, self.iterations,
                                  device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, self.iterations,
                                  device=device, dtype=x.dtype)

        # Initialize sync decay
        r_out = torch.exp(-torch.clamp(self.decay_params_out, 0, 15)) \
                     .unsqueeze(0).repeat(B, 1)
        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
            activated_state, None, None, r_out, synch_type='out')

        # Recurrent loop
        for stepi in range(self.iterations):
            # Synapse input: [syndrome, current state]
            pre_synapse_input = torch.cat((x, activated_state), dim=-1)
            state = self.synapses(pre_synapse_input)

            # Update trace
            state_trace = torch.cat(
                (state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # NLM
            activated_state = self.trace_processor(state_trace)

            # Sync for output
            synchronisation_out, decay_alpha_out, decay_beta_out = \
                self.compute_synchronisation(
                    activated_state, decay_alpha_out, decay_beta_out,
                    r_out, synch_type='out')

            # Predict
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            if track:
                pre_activations_tracking.append(
                    state_trace[:, :, -1].detach().cpu().numpy())
                post_activations_tracking.append(
                    activated_state.detach().cpu().numpy())
                synch_out_tracking.append(
                    synchronisation_out.detach().cpu().numpy())

        if track:
            return (predictions, certainties,
                    np.array(synch_out_tracking),
                    np.array(pre_activations_tracking),
                    np.array(post_activations_tracking),
                    np.array([]))
        return predictions, certainties, synchronisation_out
