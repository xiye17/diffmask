from transformers import PretrainedConfig
from transformers.utils import logging
from synth.tokenizer_synth import SYNTH_VOCAB

# for synth dataset
class SynBertConfig(PretrainedConfig):
    model_type = "roberta"

    def __init__(
        self,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=768,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_length=65,
        max_position_embeddings=65,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        gradient_checkpointing=False,
        **kwargs
    ):
        pad_token_id = SYNTH_VOCAB['<pad>']
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = len(SYNTH_VOCAB)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_length=max_length
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing

# for simple dataset
class SimBertConfig(PretrainedConfig):
    model_type = "roberta"

    def __init__(
        self,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=512,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_length=30,
        max_position_embeddings=30,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        gradient_checkpointing=False,
        **kwargs
    ):
        pad_token_id = SYNTH_VOCAB['<pad>']
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = len(SYNTH_VOCAB)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_length=max_length
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing

class CompBertConfig(PretrainedConfig):
    model_type = "roberta"

    def __init__(
        self,
        hidden_size=256,
        num_hidden_layers=3,
        num_attention_heads=8,
        intermediate_size=512,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_length=30,
        max_position_embeddings=30,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        gradient_checkpointing=False,
        **kwargs
    ):
        pad_token_id = SYNTH_VOCAB['<pad>']
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = len(SYNTH_VOCAB)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_length=max_length
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing