import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from bert.extract_features import run_bert
from nl_traj_feature_learning.nl_traj_dataset import NLTrajComparisonDataset

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
            in_features=128, out_features=128  # TODO: can decrease this to 16
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
            in_features=128, out_features=128  # TODO: decrease this to 16
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
        # TODO: take the mean over timesteps

        # BERT-encode the language
        # TODO: Make sure that we use .detach() on bert output.
        #  e.g.: run_bert(lang).detach()
        bert_output = run_bert(lang)  # TODO: use the pytorch version of BERT on HuggingFace
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


def train(nlcomp_file, traj_a_file, traj_b_file, epochs):
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load it to the specified device, either gpu or cpu
    model = NLTrajAutoencoder().to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    mse = nn.MSELoss()

    train_data = NLTrajComparisonDataset(nlcomp_file, traj_a_file, traj_b_file)
    val_dataset = None

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=128, shuffle=True  # , num_workers=4, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.to(device)
            traj_a, traj_b, lang = batch_features

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            output = model(batch_features)
            encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b = output

            # compute training reconstruction loss
            reconstruction_loss = mse(decoded_traj_a, traj_a) + mse(decoded_traj_b, traj_b)
            distance_loss = F.cosine_similarity(encoded_traj_b - encoded_traj_a, encoded_lang)
            train_loss = reconstruction_loss + distance_loss

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
