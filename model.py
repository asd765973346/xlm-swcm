import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformer import Constants
from transformers import XLMRobertaModel, XLMRobertaConfig, XLMRobertaTokenizer
from transformer.Beam import Beam


class NormalDecoderLayer(nn.Module):
    """
    Standard transformer decoder layer with self-attention, cross-attention, and feed-forward network.
    
    This layer follows the typical transformer decoder architecture with residual connections
    and layer normalization. It processes input through self-attention, cross-attention with
    encoder outputs, and a feed-forward network.
    
    Args:
        config: Configuration object containing model hyperparameters
    """
    
    def __init__(self, config):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            config.hidden_size, 
            config.num_attention_heads, 
            batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_size, 
            config.num_attention_heads, 
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, decoder_attention_mask, decoder_key_mask, 
                encoder_hidden_states, encoder_attention_mask, embed_layer):
        # Embedding and Input Processing
        hidden_states = embed_layer(input_ids)
        
        # Self-Attention Sublayer
        hidden_states = self.norm1(hidden_states)
        self_attn_output, _ = self.self_attention(
            hidden_states, 
            hidden_states, 
            hidden_states, 
            attn_mask=decoder_attention_mask, 
            key_padding_mask=decoder_key_mask,
            need_weights=False
        )
        hidden_states = hidden_states + self.dropout1(self_attn_output)
        
        # Cross-Attention Sublayer (reuse norm1)
        hidden_states = self.norm1(hidden_states)
        cross_attn_output, _ = self.cross_attention(
            hidden_states, 
            encoder_hidden_states, 
            encoder_hidden_states, 
            key_padding_mask=encoder_attention_mask
        )
        hidden_states = hidden_states + self.dropout1(cross_attn_output)

        # Feed-Forward Network
        hidden_states = self.norm2(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = hidden_states + self.dropout2(ffn_output)
        
        return hidden_states


class CustomDecoderLayer(nn.Module):
    """
    Custom decoder layer with interleaved architecture containing two FFN layers.
    
    This layer implements a modified transformer decoder with the following structure:
    1. Self-attention
    2. First FFN
    3. Cross-attention
    4. Second FFN
    
    This design allows for more complex feature transformation and interaction
    between self-attention and cross-attention mechanisms.
    
    Args:
        config: Configuration object containing model hyperparameters
    """
    
    def __init__(self, config):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            config.hidden_size, 
            config.num_attention_heads,
            batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_size, 
            config.num_attention_heads,
            batch_first=True
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Two separate FFN layers
        self.ffn1 = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        
        # Four separate layer norms for each sublayer
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.norm3 = nn.LayerNorm(config.hidden_size)
        self.norm4 = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, decoder_attention_mask, decoder_key_mask, 
                encoder_hidden_states, encoder_attention_mask, embed_layer):
        # Self-attention sublayer
        hidden_states = embed_layer(input_ids)
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        hidden_states, _ = self.self_attention(
            hidden_states, 
            hidden_states, 
            hidden_states, 
            attn_mask=decoder_attention_mask,
            key_padding_mask=decoder_key_mask,
            need_weights=False
        )
        
        # First FFN sublayer
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # Cross-attention sublayer
        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states, _ = self.cross_attention(
            hidden_states, 
            encoder_hidden_states, 
            encoder_hidden_states, 
            key_padding_mask=encoder_attention_mask
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # Second FFN sublayer
        residual = hidden_states
        hidden_states = self.norm4(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class InterleavedTransformerDecoder(nn.Module):
    """
    Interleaved transformer decoder that combines custom decoder layers with normal decoder layers.
    
    This decoder creates a hybrid architecture by:
    1. First initializing custom decoder layers using encoder weights
    2. Then inserting normal decoder layers at regular intervals
    
    The design allows for leveraging pre-trained encoder knowledge while maintaining
    the flexibility of a custom decoder architecture.
    
    Args:
        config: Configuration object containing model hyperparameters
        encoder_model: Pre-trained encoder model to initialize weights from
        embed_layer: Embedding layer to be shared
        tgtlen: Maximum target sequence length
    """
    
    def __init__(self, config, encoder_model, embed_layer, tgtlen):
        super().__init__()
        self.layers = nn.ModuleList()
        self.embed_layer = embed_layer
        self.last_linear = nn.Linear(config.hidden_size, config.vocab_size)
        self.len_max_seq = tgtlen
        
        # Initialize custom decoder layers based on encoder layers
        encoder_layers = encoder_model.encoder.layer
        for encoder_layer in encoder_layers:
            self.layers.append(CustomDecoderLayer(config))
        
        # Initialize decoder layers with encoder weights
        self.initialize_with_encoder_weights(encoder_model)
        
        # Insert normal decoder layers at regular intervals
        self.insert_normal_layers(config)
        
        print(f"Decoder initialized with {len(self.layers)} layers")

    def forward(self, input_ids, decoder_attention_mask=None, decoder_key_mask=None, 
                encoder_hidden_states=None, encoder_attention_mask=None):
        embed_layer = self.embed_layer
        hidden_states = input_ids
        
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, 
                decoder_attention_mask, 
                decoder_key_mask,
                encoder_hidden_states, 
                encoder_attention_mask, 
                embed_layer
            )
        
        logits = self.last_linear(hidden_states)
        return logits

    def initialize_with_encoder_weights(self, encoder_model):
        """
        Initialize decoder layers using encoder model weights.
        
        This method leverages pre-trained encoder weights to initialize the decoder's
        attention and feed-forward layers, potentially improving training convergence
        and model performance.
        
        Args:
            encoder_model: Pre-trained encoder model
        """
        encoder_layers = encoder_model.encoder.layer

        for index, decoder_layer in enumerate(self.layers):
            corresponding_encoder_layer = encoder_layers[index % len(encoder_layers)]
            
            # Initialize attention layers
            self._init_layer(decoder_layer.self_attention, corresponding_encoder_layer.attention.self)
            self._init_layer(decoder_layer.cross_attention, corresponding_encoder_layer.attention.self)
            
            # Initialize feed-forward networks
            self._init_ffn(decoder_layer.ffn1, corresponding_encoder_layer.intermediate, corresponding_encoder_layer.output)
            self._init_ffn(decoder_layer.ffn2, corresponding_encoder_layer.intermediate, corresponding_encoder_layer.output)

    def _init_layer(self, decoder_sublayer, encoder_sublayer):
        """Initialize decoder sublayer with encoder sublayer weights."""
        decoder_sublayer.load_state_dict(encoder_sublayer.state_dict(), strict=False)

    def _init_ffn(self, decoder_ffn, encoder_intermediate, encoder_output):
        """Initialize decoder FFN with encoder intermediate and output layer weights."""
        decoder_ffn[0].weight.data = encoder_intermediate.dense.weight.data.clone()
        decoder_ffn[0].bias.data = encoder_intermediate.dense.bias.data.clone()
        decoder_ffn[2].weight.data = encoder_output.dense.weight.data.clone()
        decoder_ffn[2].bias.data = encoder_output.dense.bias.data.clone()

    def insert_normal_layers(self, config):
        """Insert normal decoder layers at regular intervals (every 3 layers)."""
        original_layers = self.layers
        self.layers = nn.ModuleList()
        n = 3  # Insert interval
        
        for i in range(len(original_layers)):
            self.layers.append(original_layers[i])
            if (i + 1) % n == 0 and (i + 1) < len(original_layers):
                self.layers.append(NormalDecoderLayer(config))


class Seq2SeqModel(nn.Module):
    """
    Sequence-to-sequence model with XLM-RoBERTa encoder and interleaved transformer decoder.
    
    This model implements a complete encoder-decoder architecture suitable for tasks like
    text summarization, translation, and other sequence generation tasks. It features:
    - XLM-RoBERTa as the encoder for multilingual support
    - Custom interleaved decoder for enhanced generation capabilities
    - Support for both teacher forcing and autoregressive generation
    - Beam search and greedy decoding for inference
    
    Args:
        model_name_or_path: Path to pre-trained XLM-RoBERTa model
        decoder_config: Configuration for the decoder
        device: Device to run the model on
        tgtlen: Maximum target sequence length
        batchsize: Batch size
        teacher_forcing: Teacher forcing probability during training
    """
    
    def __init__(self, model_name_or_path, decoder_config, device, tgtlen, batchsize, teacher_forcing):
        super().__init__()
        self.device = device
        self.config = decoder_config
        self.batchsize = batchsize
        self.tgtlen = tgtlen
        self.teacher_forcing = teacher_forcing
        self.len_max_seq = tgtlen
        
        # Initialize encoder and decoder
        self.encoder = XLMRobertaModel.from_pretrained(model_name_or_path)
        self.embed_layer = self.encoder.embeddings
        self.decoder = InterleavedTransformerDecoder(
            decoder_config, self.encoder, self.embed_layer, tgtlen
        )

    def forward(self, input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask):
        """
        Forward pass with support for both teacher forcing and autoregressive generation.
        
        Args:
            input_ids: Input token IDs for encoder
            encoder_attention_mask: Attention mask for encoder
            decoder_input_ids: Input token IDs for decoder
            decoder_attention_mask: Attention mask for decoder
            
        Returns:
            logits: Output logits for next token prediction
        """
        # Encode input sequence
        encoder_outputs = self.encoder(input_ids)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # Generate decoder attention masks
        seq_len = decoder_input_ids.size(1)
        batch_size = decoder_input_ids.size(0)
        n_head = self.config.num_attention_heads
        device = decoder_input_ids.device

        # Create causal mask for self-attention
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        expanded_causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        expanded_causal_mask = expanded_causal_mask.unsqueeze(0).expand(n_head, -1, -1, -1)
        decoder_causal_mask = expanded_causal_mask.reshape(n_head * batch_size, seq_len, seq_len)

        # Teacher forcing vs autoregressive generation
        if self.training and random.random() < self.teacher_forcing:
            # Use teacher forcing
            logits = self.decoder(
                decoder_input_ids,
                decoder_attention_mask=decoder_causal_mask,
                decoder_key_mask=~decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=~encoder_attention_mask
            )
        else:
            # Autoregressive generation
            outputs = []
            decoder_input = decoder_input_ids[:, 0].unsqueeze(1)  # Start with BOS token
            
            for i in range(self.tgtlen):
                seq_len = decoder_input.size(1)
                batch_size = decoder_input.size(0)
                n_head = self.config.num_attention_heads
                
                # Generate causal mask for current step
                causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                causal_mask = causal_mask.unsqueeze(0).repeat(n_head * batch_size, 1, 1)
                causal_mask = causal_mask.to(self.device)
                
                step_logits = self.decoder(
                    decoder_input,
                    decoder_attention_mask=causal_mask,
                    decoder_key_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=~encoder_attention_mask
                )
                
                outputs.append(step_logits[:, -1:, :])
                
                if i < self.tgtlen - 1:
                    next_token = step_logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            logits = torch.cat(outputs, dim=1)
        
        return logits
    
    def beam_decode(self, src_seq, src_mask, beam_size, n_best):
        """
        Beam search decoding for generating high-quality sequences.
        
        Args:
            src_seq: Source sequence tokens
            src_mask: Source sequence attention mask
            beam_size: Number of beams to maintain
            n_best: Number of best sequences to return
            
        Returns:
            batch_hyp: Generated hypothesis sequences
            batch_scores: Scores for each hypothesis
        """
        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, beam_size):
            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * beam_size, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(src_seq, src_enc, src_mask, inst_idx_to_position_map, active_inst_idx_list):
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, beam_size)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, beam_size)
            active_src_mask = collect_active_part(src_mask, active_inst_idx, n_prev_active_inst, beam_size)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_src_mask, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, enc_output, src_mask, inst_idx_to_position_map, beam_size):
            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def predict_word(dec_seq, src_seq, enc_output, src_mask, n_active_inst, beam_size):
                seq_len = dec_seq.size(1)
                n_head = self.config.num_attention_heads
                
                # Create causal mask
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device), diagonal=1)
                causal_mask = causal_mask.unsqueeze(0).expand(n_active_inst * beam_size, -1, -1)
                causal_mask = causal_mask.unsqueeze(0).expand(n_head, -1, -1, -1)
                causal_mask = causal_mask.reshape(n_head * n_active_inst * beam_size, seq_len, seq_len)

                dec_output = self.decoder(
                    dec_seq, 
                    decoder_attention_mask=causal_mask,
                    decoder_key_mask=None,
                    encoder_hidden_states=enc_output,
                    encoder_attention_mask=src_mask
                )
                dec_output = dec_output[:, -1, :]
                word_prob = F.log_softmax(dec_output, dim=1)
                word_prob = word_prob.view(n_active_inst, beam_size, -1)

                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            word_prob = predict_word(dec_seq, src_seq, enc_output, src_mask, n_active_inst, beam_size)

            active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            src_enc = self.encoder(input_ids=src_seq, attention_mask=src_mask)[0]

            n_inst, len_s, d_h = src_enc.size()
            src_seq = src_seq.repeat(1, beam_size).view(n_inst * beam_size, len_s)
            src_enc = src_enc.repeat(1, beam_size, 1).view(n_inst * beam_size, len_s, d_h)
            src_mask = src_mask.repeat(1, beam_size).view(n_inst * beam_size, len_s)

            inst_dec_beams = [Beam(beam_size, device=self.device) for _ in range(n_inst)]

            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            for len_dec_seq in range(1, self.decoder.len_max_seq + 1):
                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, src_seq, src_enc, src_mask, inst_idx_to_position_map, beam_size)

                if not active_inst_idx_list:
                    break

                src_seq, src_enc, src_mask, inst_idx_to_position_map = collate_active_info(
                    src_seq, src_enc, src_mask, inst_idx_to_position_map, active_inst_idx_list)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_best)

        return batch_hyp, batch_scores

    def greedy_decode(self, src_seq, src_mask):
        """
        Greedy decoding for fast inference.
        
        Args:
            src_seq: Source sequence tokens
            src_mask: Source sequence attention mask
            
        Returns:
            dec_seq: Generated sequence
        """
        enc_output = self.encoder(input_ids=src_seq, attention_mask=src_mask)[0]
        dec_seq = torch.full((src_seq.size(0),), Constants.BOS).unsqueeze(-1).type_as(src_seq)

        for i in range(self.len_max_seq):
            dec_output = self.decoder(
                dec_seq, 
                decoder_attention_mask=None,
                encoder_hidden_states=enc_output, 
                encoder_attention_mask=src_mask
            )
            dec_output = dec_output[:, -1, :].max(-1)[1]
            dec_seq = torch.cat((dec_seq, dec_output.unsqueeze(-1)), 1)
        
        return dec_seq