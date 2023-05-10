import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import bert.extract_features as b
from nl_traj_feature_learning.nl_traj_dataset import NLTrajComparisonDataset
import argparse
import os

STATE_DIM = 64
ACTION_DIM = 4  # NOTE: we use OSC_POSITION as our controller
BERT_OUTPUT_DIM = 768


class NLTrajAutoencoder (nn.Module):
    def __init__(self, encoder_hidden_dim=128, decoder_hidden_dim=128, preprocessed_nlcomps=False, **kwargs):
        super().__init__()
        # TODO: can later make encoders and decoders transformers
        self.traj_encoder_hidden_layer = nn.Linear(
            in_features=STATE_DIM+ACTION_DIM, out_features=encoder_hidden_dim
        )
        self.traj_encoder_output_layer = nn.Linear(
            in_features=encoder_hidden_dim, out_features=16
        )
        self.traj_decoder_hidden_layer = nn.Linear(
            in_features=16, out_features=decoder_hidden_dim
        )
        self.traj_decoder_output_layer = nn.Linear(
            in_features=decoder_hidden_dim, out_features=STATE_DIM+ACTION_DIM
        )

        self.preprocessed_nlcomps = preprocessed_nlcomps
        # Note: the first language encoder layer is BERT.
        self.lang_encoder_hidden_layer = nn.Linear(
            in_features=BERT_OUTPUT_DIM, out_features=encoder_hidden_dim
        )
        self.lang_encoder_output_layer = nn.Linear(
            in_features=encoder_hidden_dim, out_features=16
        )
        self.lang_decoder_output_layer = None  # TODO: implement language decoder later.

    # Input is a tuple with (trajectory_a, trajectory_b, language)
    # traj_a has shape (n_trajs, n_timesteps, state+action)
    def forward(self, input):
        traj_a = input[0]
        traj_b = input[1]
        lang = input[2]

        # Encode trajectories
        encoded_traj_a = self.traj_encoder_output_layer(torch.relu(self.traj_encoder_hidden_layer(traj_a)))
        encoded_traj_b = self.traj_encoder_output_layer(torch.relu(self.traj_encoder_hidden_layer(traj_b)))
        # Take the mean over timesteps
        encoded_traj_a = torch.mean(encoded_traj_a, dim=-2)
        encoded_traj_b = torch.mean(encoded_traj_b, dim=-2)

        if not self.preprocessed_nlcomps:
            # print("RUNNING BERT...")
            # BERT-encode the language
            # TODO: Make sure that we use .detach() on bert output.
            #  e.g.: run_bert(lang).detach()
            bert_input = ""
            for l in lang:
                bert_input = bert_input + l + "\n"

            bert_output = b.run_bert(bert_input)
            bert_output_embeddings = []
            # Loop over the batch
            for i, l in enumerate(lang):
                bert_output_words = bert_output[i]['features']
                bert_output_embedding = []
                for word_embedding in bert_output_words:
                    bert_output_embedding.append(word_embedding['layers'][0]['values'])
                # NOTE: We average across timesteps (since BERT produces a per-token embedding).
                bert_output_embedding = np.mean(np.asarray(bert_output_embedding), axis=0)
                print("bert_output_embedding:", bert_output_embedding.shape)
                bert_output_embeddings.append(bert_output_embedding)
            bert_output_embeddings = np.asarray(bert_output_embeddings)
            bert_output_embeddings = torch.as_tensor(bert_output_embeddings, dtype=torch.float32)
            print("bert_output_embeddings:", bert_output_embeddings.shape)

            # NOTE: The following code doesn't pass lang in as a batch, and is thus VERY slow.
            # bert_output_embeddings = []
            # for l in lang:
            #     bert_output = b.run_bert(l)  # TODO: use the pytorch version of BERT on HuggingFace (is this necessary, since lang isn't a tensor?)
            #     bert_output_words = bert_output[0]['features']
            #     bert_output_embedding = []
            #     for word_embedding in bert_output_words:
            #         bert_output_embedding.append(word_embedding['layers'][0]['values'])
            #     # NOTE: We average across timesteps (since BERT produces a per-token embedding).
            #     bert_output_embedding = np.mean(np.asarray(bert_output_embedding), axis=0)
            #     print("bert_output_embedding:", bert_output_embedding.shape)
            #     bert_output_embeddings.append(bert_output_embedding)
            # bert_output_embeddings = np.asarray(bert_output_embeddings)
            # bert_output_embeddings = torch.as_tensor(bert_output_embeddings, dtype=torch.float32)
            # print("bert_output_embeddings:", bert_output_embeddings.shape)
        else:
            # print("USING PRE-COMPUTED BERT EMBEDDINGS...")
            bert_output_embeddings = lang

        # Encode the language
        encoded_lang = self.lang_encoder_output_layer(torch.relu(self.lang_encoder_hidden_layer(bert_output_embeddings)))

        # NOTE: traj_a is the reference, traj_b is the updated

        decoded_traj_a = self.traj_decoder_output_layer(torch.relu(self.traj_decoder_hidden_layer(encoded_traj_a)))
        decoded_traj_b = self.traj_decoder_output_layer(torch.relu(self.traj_decoder_hidden_layer(encoded_traj_b)))

        output = (encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b)
        return output


def train(seed, data_dir, epochs, save_dir, encoder_hidden_dim=128, decoder_hidden_dim=128, preprocessed_nlcomps=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # load it to the specified device, either gpu or cpu
    print("Initializing model and loading to device...")
    model = NLTrajAutoencoder(encoder_hidden_dim, decoder_hidden_dim, preprocessed_nlcomps).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    mse = nn.MSELoss()
    logsigmoid = nn.LogSigmoid()

    print("Loading dataset...")

    # Some file-handling logic first.
    if preprocessed_nlcomps:
        train_nlcomp_file = os.path.join(data_dir, "train/nlcomps.npy")
        val_nlcomp_file = os.path.join(data_dir, "val/nlcomps.npy")
    else:
        train_nlcomp_file = os.path.join(data_dir, "train/nlcomps.json")
        val_nlcomp_file = os.path.join(data_dir, "val/nlcomps.json")
    train_traj_a_file = os.path.join(data_dir, "train/traj_as.npy")
    train_traj_b_file = os.path.join(data_dir, "train/traj_bs.npy")
    val_traj_a_file = os.path.join(data_dir, "val/traj_as.npy")
    val_traj_b_file = os.path.join(data_dir, "val/traj_bs.npy")

    train_dataset = NLTrajComparisonDataset(train_nlcomp_file, train_traj_a_file, train_traj_b_file, preprocessed_nlcomps=preprocessed_nlcomps)
    val_dataset = NLTrajComparisonDataset(val_nlcomp_file, val_traj_a_file, val_traj_b_file, preprocessed_nlcomps=preprocessed_nlcomps)

    # NOTE: this creates a dataset that doesn't have trajectories separated across datasets. DEPRECATED.
    # generator = torch.Generator().manual_seed(seed)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[0.9, 0.1], generator=generator)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    print("Initial loss sanity check:")
    temp_loss = 0
    for train_datapoint in train_loader:
        with torch.no_grad():
            traj_a, traj_b, lang = train_datapoint
            traj_a = torch.as_tensor(traj_a, dtype=torch.float32, device=device)
            traj_b = torch.as_tensor(traj_b, dtype=torch.float32, device=device)
            if preprocessed_nlcomps:
                lang = torch.as_tensor(lang, dtype=torch.float32, device=device)
            # lang = torch.as_tensor(lang, device=device)
            train_datapoint = (traj_a, traj_b, lang)
            pred = model(train_datapoint)

            encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b = pred
            reconstruction_loss = mse(decoded_traj_a, torch.mean(traj_a, dim=-2)) + mse(decoded_traj_b,
                                                                                        torch.mean(traj_b, dim=-2))

            dot_prod = torch.einsum('ij,ij->i', encoded_traj_b - encoded_traj_a, encoded_lang)
            log_likelihood = torch.mean(logsigmoid(dot_prod))
            log_likelihood_loss = -1 * log_likelihood

            temp_loss += (reconstruction_loss + log_likelihood_loss).item()
    temp_loss /= len(train_loader)
    print("initial train loss:", temp_loss)
    temp_loss = 0
    for val_datapoint in val_loader:
        with torch.no_grad():
            traj_a, traj_b, lang = val_datapoint
            traj_a = torch.as_tensor(traj_a, dtype=torch.float32, device=device)
            traj_b = torch.as_tensor(traj_b, dtype=torch.float32, device=device)
            if preprocessed_nlcomps:
                lang = torch.as_tensor(lang, dtype=torch.float32, device=device)
            # lang = torch.as_tensor(lang, device=device)
            val_datapoint = (traj_a, traj_b, lang)
            pred = model(val_datapoint)

            encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b = pred
            reconstruction_loss = mse(decoded_traj_a, torch.mean(traj_a, dim=-2)) + mse(decoded_traj_b,
                                                                                        torch.mean(traj_b, dim=-2))

            dot_prod = torch.einsum('ij,ij->i', encoded_traj_b - encoded_traj_a, encoded_lang)
            log_likelihood = torch.mean(logsigmoid(dot_prod))
            log_likelihood_loss = -1 * log_likelihood

            temp_loss += (reconstruction_loss + log_likelihood_loss).item()
    temp_loss /= len(val_loader)
    print("initial val loss:", temp_loss)

    print("Beginning training...")
    train_losses = []
    val_losses = []
    val_reconstruction_losses = []
    val_cosine_similarities = []
    val_log_likelihoods = []
    accuracies = []
    for epoch in range(epochs):
        loss = 0
        for train_datapoint in train_loader:
            traj_a, traj_b, lang = train_datapoint

            # load it to the active device
            # also cast down (from float64 in np) to float32, since PyTorch's matrices are float32.
            traj_a = torch.as_tensor(traj_a, dtype=torch.float32, device=device)
            traj_b = torch.as_tensor(traj_b, dtype=torch.float32, device=device)
            if preprocessed_nlcomps:
                lang = torch.as_tensor(lang, dtype=torch.float32, device=device)
            # lang = torch.as_tensor(lang, device=device)

            # train_datapoint = train_datapoint.to(device)  # Shouldn't be needed, since already on device
            train_datapoint = (traj_a, traj_b, lang)

            # reset the gradients back to zero
            optimizer.zero_grad()

            # compute reconstructions
            output = model(train_datapoint)
            encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b = output

            # compute training reconstruction loss
            # MSELoss already takes the mean over the batch.
            reconstruction_loss = mse(decoded_traj_a, torch.mean(traj_a, dim=-2)) + mse(decoded_traj_b, torch.mean(traj_b, dim=-2))
            # print("reconstruction_loss:", reconstruction_loss.shape)

            # F.cosine_similarity only reduces along the feature dimension, so we take the mean over the batch later.
            cos_sim = F.cosine_similarity(encoded_traj_b - encoded_traj_a, encoded_lang)
            cos_sim = torch.mean(cos_sim)  # Take the mean over the batch.
            distance_loss = 1 - cos_sim  # Then convert the value to a loss.
            # print("distance_loss:", distance_loss.shape)

            dot_prod = torch.einsum('ij,ij->i', encoded_traj_b - encoded_traj_a, encoded_lang)
            log_likelihood = logsigmoid(dot_prod)
            log_likelihood = torch.mean(log_likelihood)  # Take the mean over the batch.
            log_likelihood_loss = -1 * log_likelihood  # Then convert the value to a loss.

            # By now, train_loss is a scalar.
            # train_loss = reconstruction_loss + distance_loss
            train_loss = reconstruction_loss + log_likelihood_loss
            # print("train_loss:", train_loss.shape)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        # Note: this is the per-BATCH loss. len(train_loader) gives number of batches.
        loss = loss / len(train_loader)

        # Evaluation
        val_loss = 0
        val_reconstruction_loss = 0
        val_cosine_similarity = 0
        val_log_likelihood = 0
        num_correct = 0
        for val_datapoint in val_loader:
            with torch.no_grad():
                traj_a, traj_b, lang = val_datapoint
                traj_a = torch.as_tensor(traj_a, dtype=torch.float32, device=device)
                traj_b = torch.as_tensor(traj_b, dtype=torch.float32, device=device)
                if preprocessed_nlcomps:
                    lang = torch.as_tensor(lang, dtype=torch.float32, device=device)
                # lang = torch.as_tensor(lang, device=device)
                val_datapoint = (traj_a, traj_b, lang)
                pred = model(val_datapoint)

                encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b = pred
                reconstruction_loss = mse(decoded_traj_a, torch.mean(traj_a, dim=-2)) + mse(decoded_traj_b, torch.mean(traj_b, dim=-2))
                val_reconstruction_loss += reconstruction_loss.item()  # record

                cos_sim = torch.mean(F.cosine_similarity(encoded_traj_b - encoded_traj_a, encoded_lang))
                distance_loss = 1 - cos_sim
                val_cosine_similarity += cos_sim.item()

                dot_prod = torch.einsum('ij,ij->i', encoded_traj_b - encoded_traj_a, encoded_lang)
                log_likelihood = torch.mean(logsigmoid(dot_prod))
                log_likelihood_loss = -1 * log_likelihood
                val_log_likelihood += log_likelihood.item()

                val_loss += (reconstruction_loss + log_likelihood_loss).item()
                num_correct += np.sum(dot_prod.detach().cpu().numpy() > 0)

        val_loss /= len(val_loader)
        val_reconstruction_loss /= len(val_loader)
        val_cosine_similarity /= len(val_loader)
        val_log_likelihood /= len(val_loader)
        accuracy = num_correct / len(val_dataset)

        # display the epoch training loss
        print("epoch : {}/{}, [train] loss = {:.6f}, [val] loss = {:.6f}, [val] reconstruction_loss = {:.6f}, [val] cosine_similarity = {:.6f}, [val] log_likelihood = {:.6f}, [val] accuracy = {:.6f}".format(epoch + 1, epochs, loss, val_loss, val_reconstruction_loss, val_cosine_similarity, val_log_likelihood, accuracy))
        # Don't forget to save the model (as we go)!
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_state_dict.pth'))
        torch.save(model, os.path.join(save_dir, 'model.pth'))
        train_losses.append(loss)
        val_losses.append(val_loss)
        val_reconstruction_losses.append(val_reconstruction_loss)
        val_cosine_similarities.append(val_cosine_similarity)
        val_log_likelihoods.append(val_log_likelihood)
        accuracies.append(accuracy)
        np.save(os.path.join(save_dir, 'train_losses.npy'), np.asarray(train_losses))
        np.save(os.path.join(save_dir, 'val_losses.npy'), np.asarray(val_losses))
        np.save(os.path.join(save_dir, 'val_reconstruction_losses.npy'), np.asarray(val_reconstruction_losses))
        np.save(os.path.join(save_dir, 'val_cosine_similarities.npy'), np.asarray(val_cosine_similarities))
        np.save(os.path.join(save_dir, 'val_log_likelihoods.npy'), np.asarray(val_log_likelihoods))
        np.save(os.path.join(save_dir, 'accuracies.npy'), np.asarray(accuracies))


    # # Evaluation
    # num_correct = 0
    # log_likelihood = 0
    # for val_datapoint in val_loader:
    #     with torch.no_grad():
    #         traj_a, traj_b, lang = val_datapoint
    #         traj_a = torch.as_tensor(traj_a, dtype=torch.float32, device=device)
    #         traj_b = torch.as_tensor(traj_b, dtype=torch.float32, device=device)
    #         # lang = torch.as_tensor(lang, device=device)
    #
    #         val_datapoint = (traj_a, traj_b, lang)
    #         pred = model(val_datapoint)
    #         encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b = pred
    #
    #         encoded_traj_a = encoded_traj_a.detach().cpu().numpy()
    #         encoded_traj_b = encoded_traj_b.detach().cpu().numpy()
    #         encoded_lang = encoded_lang.detach().cpu().numpy()
    #
    #         dot_prod = np.dot(encoded_traj_b-encoded_traj_a, encoded_lang)
    #         if dot_prod > 0:
    #             num_correct += 1
    #         log_likelihood += np.log(1/(1 + np.exp(-dot_prod)))
    #
    # accuracy = num_correct / len(val_loader)
    # print("final accuracy:", accuracy)
    # print("final log likelihood:", log_likelihood)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--data-dir', type=str, default='', help='')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--save-dir', type=str, default='', help='')
    parser.add_argument('--encoder-hidden-dim', type=int, default=128, help='')
    parser.add_argument('--decoder-hidden-dim', type=int, default=128, help='')
    parser.add_argument('--preprocessed-nlcomps', action="store_true", help='')

    args = parser.parse_args()

    trained_model = train(args.seed, args.data_dir, args.epochs, args.save_dir,
                          encoder_hidden_dim=args.encoder_hidden_dim, decoder_hidden_dim=args.decoder_hidden_dim,
                          preprocessed_nlcomps=args.preprocessed_nlcomps)

