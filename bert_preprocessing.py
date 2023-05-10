import json
import numpy as np
import bert.extract_features as b
import argparse
import os


def preprocess_strings(nlcomp_dir, batch_size):
    # nlcomp_file is a json file with the list of comparisons in NL.
    nlcomp_file = os.path.join(nlcomp_dir, 'nlcomps.json')

    with open(nlcomp_file, 'rb') as f:
        nlcomps = json.load(f)

    batches = []
    for i, l in enumerate(nlcomps):
        if i % batch_size == 0:
            batches.append("")
        batches[-1] = batches[-1] + l + "\n"

    bert_output_embeddings = []
    for bert_input in batches:
        bert_output = b.run_bert(bert_input)

        # Loop over the batch
        for i in range(len(bert_output)):
            bert_output_words = bert_output[i]['features']
            bert_output_embedding = []
            for word_embedding in bert_output_words:
                bert_output_embedding.append(word_embedding['layers'][0]['values'])
            # NOTE: We average across timesteps (since BERT produces a per-token embedding).
            bert_output_embedding = np.mean(np.asarray(bert_output_embedding), axis=0)
            # print("bert_output_embedding:", bert_output_embedding.shape)
            bert_output_embeddings.append(bert_output_embedding)

        outfile = os.path.join(nlcomp_dir, 'nlcomps.npy')
        np.save(outfile, np.asarray(bert_output_embeddings))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--nlcomp-dir', type=str, default='', help='')
    parser.add_argument('--batch-size', type=int, default=5000, help='')

    args = parser.parse_args()
    preprocess_strings(args.nlcomp_dir, args.batch_size)

