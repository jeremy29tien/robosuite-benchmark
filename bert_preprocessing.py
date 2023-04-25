import json
import numpy as np
import bert.extract_features as b


nlcomp_file = '/home/jeremy/robosuite-benchmark/data/nl-traj/all-pairs/nlcomps.json'
# nlcomp_file is a json file with the list of comparisons in NL.
with open(nlcomp_file, 'rb') as f:
    nlcomps = json.load(f)

bert_input = ""
batches = []
batch_size = 5000
for i, l in enumerate(nlcomps):
    if i % batch_size == 0:
        batches.append("")
    batches[-1] = batches[-1] + l + "\n"
    # bert_input = bert_input + l + "\n"

bert_outputs = []
for bert_input in batches:
    bert_outputs.append(b.run_bert(bert_input))
# bert_output = b.run_bert(bert_input)

bert_output_embeddings = []
# Loop over the batch
for i, l in enumerate(nlcomps):
    bert_output_words = bert_outputs[i // batch_size][i % batch_size]['features']  # bert_outputs[i]['features']
    bert_output_embedding = []
    for word_embedding in bert_output_words:
        bert_output_embedding.append(word_embedding['layers'][0]['values'])
    # NOTE: We average across timesteps (since BERT produces a per-token embedding).
    bert_output_embedding = np.mean(np.asarray(bert_output_embedding), axis=0)
    # print("bert_output_embedding:", bert_output_embedding.shape)
    bert_output_embeddings.append(bert_output_embedding)
bert_output_embeddings = np.asarray(bert_output_embeddings)
print("bert_output_embeddings:", bert_output_embeddings.shape)

np.save('/home/jeremy/robosuite-benchmark/data/nl-traj/all-pairs/nlcomps.npy', bert_output_embeddings)
