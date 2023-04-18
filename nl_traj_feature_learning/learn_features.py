import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bert.extract_features import run_bert

STATE_DIM = 64
ACTION_DIM = 8
BERT_OUTPUT_DIM = 768


class NLTrajAutoencoder (nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # TODO: can later make encoders and decoders transformers
        self.traj_encoder_hidden_layer = nn.Linear(
            in_features=STATE_DIM+ACTION_DIM, out_features=128
        )
        self.traj_encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.traj_decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.traj_decoder_output_layer = nn.Linear(
            in_features=128, out_features=STATE_DIM+ACTION_DIM
        )

        # Note: the first language encoder layer is BERT.
        self.lang_encoder_hidden_layer = nn.Linear(
            in_features=BERT_OUTPUT_DIM, out_features=128
        )
        self.lang_encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.lang_decoder_output_layer = None  # TODO: implement language decoder later.

    # Input is a tuple with (trajectory_a, trajectory_b, language)
    def forward(self, input):
        traj_a = input[0]
        traj_b = input[1]
        lang = input[2]

        # Encode trajectories
        encoded_traj_a = self.traj_encoder_output_layer(torch.relu(self.traj_encoder_hidden_layer(traj_a)))
        encoded_traj_b = self.traj_encoder_output_layer(torch.relu(self.traj_encoder_hidden_layer(traj_b)))

        # BERT-encode the language
        bert_output = run_bert(lang)
        bert_output_words = bert_output[0]['features']
        bert_output_embedding = []
        for word_embedding in bert_output_embedding:
            bert_output_embedding.append(word_embedding['layers'][0]['values'])
        # NOTE: We average across timesteps (since BERT produces a per-token embedding).
        bert_output_embedding = np.mean(np.asarray(bert_output_embedding), axis=0)

        # Encode the language
        encoded_lang = self.lang_encoder_output_layer(torch.relu(self.lang_encoder_hidden_layer(bert_output_embedding)))

        # NOTE: traj_a is the reference, traj_b is the updated
        # NOTE: We won't use this distance; we'll compute it during training.
        distance = F.cosine_similarity(encoded_traj_b - encoded_traj_a, encoded_lang)

        decoded_traj_a = self.traj_decoder_output_layer(torch.relu(self.traj_decoder_hidden_layer(encoded_traj_a)))
        decoded_traj_b = self.traj_decoder_output_layer(torch.relu(self.traj_decoder_hidden_layer(encoded_traj_b)))

        output = (encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b)
        return output
