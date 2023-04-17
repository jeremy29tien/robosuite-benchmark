import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import bert.extract_features

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
        self.lang_encoder_output_layer = nn.Linear(
            in_features=BERT_OUTPUT_DIM, out_features=128
        )
        self.lang_decoder_output_layer = None  # TODO: implement language decoder later.

    # Input is a tuple with (trajectory_a, trajectory_b, language)
    def forward(self, input):
        traj_a = input[0]
        traj_b = input[1]
        lang = input[2]

        encoded_traj_a = self.traj_encoder_output_layer(torch.relu(self.traj_encoder_hidden_layer(traj_a)))
        encoded_traj_b = self.traj_encoder_output_layer(torch.relu(self.traj_encoder_hidden_layer(traj_b)))
        encoded_lang = self.lang_encoder_output_layer(torch.relu(self.run_bert(lang)))

        # NOTE: traj_a is the reference, traj_b is the updated
        # NOTE: We won't use this distance; we'll compute it during training.
        distance = F.cosine_similarity(encoded_traj_b - encoded_traj_a, encoded_lang)

        decoded_traj_a = self.traj_decoder_output_layer(torch.relu(self.traj_decoder_hidden_layer(encoded_traj_a)))
        decoded_traj_b = self.traj_decoder_output_layer(torch.relu(self.traj_decoder_hidden_layer(encoded_traj_b)))

        output = (encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b)
        return output

    def run_bert(self, nl_input):
        tf.logging.set_verbosity(tf.logging.INFO)

        layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

        is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.compat.v1.estimator.tpu.RunConfig(
            master=FLAGS.master,
            tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
                num_shards=FLAGS.num_tpu_cores,
                per_host_input_for_training=is_per_host))

        examples = read_examples(FLAGS.input_file)

        features = convert_examples_to_features(
            examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)

        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature

        model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=FLAGS.init_checkpoint,
            layer_indexes=layer_indexes,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            predict_batch_size=FLAGS.batch_size)

        input_fn = input_fn_builder(
            features=features, seq_length=FLAGS.max_seq_length)

        outputs = []
        # with codecs.getwriter("utf-8")(tf.gfile.Open(FLAGS.output_file,
        #                                              "w")) as writer:
        for result in estimator.predict(input_fn, yield_single_examples=True):
            unique_id = int(result["unique_id"])
            feature = unique_id_to_feature[unique_id]
            output_json = collections.OrderedDict()
            output_json["linex_index"] = unique_id
            all_features = []
            for (i, token) in enumerate(feature.tokens):
                all_layers = []
                for (j, layer_index) in enumerate(layer_indexes):
                    layer_output = result["layer_output_%d" % j]
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    layers["values"] = [
                        round(float(x), 6) for x in layer_output[i:(i + 1)].flat
                    ]
                    all_layers.append(layers)
                features = collections.OrderedDict()
                features["token"] = token
                features["layers"] = all_layers
                all_features.append(features)
            output_json["features"] = all_features
            # writer.write(json.dumps(output_json) + "\n")
            outputs.append(output_json)
        return outputs


    def traj_encoder():
        pass