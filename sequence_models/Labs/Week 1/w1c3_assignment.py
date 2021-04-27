import os 
import random as rnd 
import trax 

# set random seeds to make this notebook easier to replicate
trax.supervised.training.init_host_and_devices(random_seed=32)

import trax.fastmath.numpy as np 
from trax import layers as tl 
from utils import Layer, load_tweets, process_tweet 

a = np.array(5.0)
print(a)
print(type(a))

def f(x):
    return x**2 

print(f"f(a) for a = 2 is {f(a)}")

grad_f = trax.fastmath.grad(fun=f)
print(type(grad_f))

grad_calculation = grad_f(a)
print(grad_calculation)

import numpy as np 
all_positive_tweets, all_negative_tweets = load_tweets()
print(f"The number of positive tweets: {len(all_positive_tweets)}")
print(f"The number of negative tweets: {len(all_negative_tweets)}")

# split positve set into validatin and training
val_pos = all_positive_tweets[4000:] 
train_pos = all_positive_tweets[:4000]
val_neg = all_negative_tweets[4000:] 
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg 
val_x = val_pos + val_neg 

train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
val_y = np.append(np.ones(len(val_pos)),  np.zeros(len(val_neg)))

print(f"length of train_x {len(train_x)}")
print(f"length of val_x {len(val_x)}")

print("Original tweet at training position 0")
print(train_pos[0])

print("Tweet at training position 0 after processing")
process_tweet(train_pos[0])

# building the vocabulary
Vocab = {'__pad': 0, '__</e>__': 1, '__UNK__': 2}
for tweet in train_x:
    processed_tweet = process_tweet(tweet)
    for word in processed_tweet:
        if word not in Vocab:
            Vocab[word] = len(Vocab)

print("Total words in vocab are", len(Vocab))
print(Vocab)

# converting a tweet to a tensor 
def tweet_to_tensor(tweet, vocab_dict, unk_token = '__UNK__', verbose = False):
    tensor = []
    processed = process_tweet(tweet)
    if verbose:
        print("List of words from the processed tweet")
        print(processed)

    unk_id = vocab_dict.get(unk_token)
    if verbose:
        print("The unique integer id for the unk_token is", unk_id)
    for word in processed:
        tensor.append(vocab_dict.get(word, unk_token))

    return tensor

# implement data generator
def data_generator(data_pos, data_neg, batch_size, loop, vocab_dict, shuffle=False):
    assert batch_size % 2 == 0 
    n_to_take = batch_size // 2 

    pos_index = 0 
    neg_index = 0 
    
    len_data_pos = len(data_pos)
    len_data_neg = len(data_neg)

    pos_index_lines = [*range(len_data_pos)]
    neg_index_lines = [*range(len_data_neg)]

    if shuffle:
        rnd.shuffle(pos_index_lines)
        rnd.shuffle(neg_index_lines)

    stop = False
    while not stop:
        batch = []
        for i in range(n_to_take):
            if pos_index >= len_data_pos:
                if not loop:
                    stop = True 
                    break
                pos_index = 0 
                if shuffle:
                    rnd.shuffle(pos_index_lines)

            tweet = data_pos[pos_index_lines[pos_index]]
            tensor = tweet_to_tensor(tweet, vocab_dict)
            batch.append(tensor)
            pos_index += 1 

        for i in range(n_to_take):
            if neg_index >= len_data_neg:
                if not loop:
                    stop = True
                    break 
                neg_index = 0 
                if shuffle:
                    rnd.shuffle(neg_index_lines)

            tweet = data_neg[neg_index_lines[neg_index]]
            tensor = tweet_to_tensor(tweet, vocab_dict)
            batch.append(tensor)
            neg_index += 1 

        if stop:
            break 

        pos_index += n_to_take 
        neg_index += n_to_take
        max_len = max([len(t) for t in batch])

        tensor_pad = []
        for tensor in batch:
            n_pad = max_len - len(tensor)
            pad = [0]*n_pad
            tensor_pad.append(tensor + pad) 

        inputs = np.array(tensor_pad)
        target_pos = [1]*n_to_take
        target_neg = [0]*n_to_take

        target_l = target_pos + target_neg

        targets = np.array(target_l)
        example_weights = np.ones_like(targets)

        yield inputs, targets, example_weights
            

rnd.seed(30)

# create the training data generator 
def train_generator(batch_size, shuffle=False):
    return data_generator(train_pos, train_neg, batch_size, True, Vocab, shuffle)

def val_generator(batch_size, shuffle=False):
    return data_generator(train_pos, train_neg, batch_size, True, Vocab, shuffle)

def test_generator(batch_size, shuffle=False):
    return data_generator(train_pos, train_neg, batch_size, False, Vocab, shuffle)

inputs, targets, example_weights = next(train_generator(4, shuffle=True))

print("Inputs: ", inputs)
print("Targets: ", targets)
print("Example Weights: ", example_weights)

class Relu(Layer):
    def forward(self, x):
        activation = np.maximum(x, 0)
        return activation

x = np.array([[-2.0, -1.0, 0.0], [0.0, 1.0, 2.0]], dtype=float)
relu_layer = Relu()
print(relu_layer(x))

from trax import fastmath
np = fastmath.numpy
random = fastmath.random

tmp_key = fastmath.random.get_prng(seed=1)
print("The random seed generated by random.get_prng")
print(tmp_key)

tmp_shape = (2, 3)
print(tmp_shape)

tmp_weight = trax.fastmath.random.normal(key=tmp_key, shape=tmp_shape)
print(tmp_weight)

class Dense(Layer):
    def __init__(self, n_units, init_stdev=0.1):
        self._n_units = n_units
        self._init_stdev = init_stdev

    def forward(self, x):
        dense = np.dot(x, self.weights)
        return dense

    def init_weights_and_state(self, input_signature, random_key):
        input_shape = input_signature.shape
        w = self._init_stdev * random.normal(key = random_key, shape = (input_shape[-1], self._n_units))
        self.weights = w
        return self.weights

dense_layer = Dense(n_units = 10)
random_key = random.get_prng(seed=0)
z = np.array([[2.0, 7.0, 25.0]])
dense_layer.init(z, random_key)
print(dense_layer.weights)
        

def classifier(vocab_size=len(Vocab), embedding_dim=256, output_dim=2, mode='train'):
    embed_layer = tl.Embedding(vocab_size=vocab_size, d_feature=embedding_dim)
    mean_layer = tl.Mean(axis=1)
    dense_output_layer = tl.Dense(n_units=output_dim)
    log_softmax_layer = tl.LogSoftmax()

    model = tl.Serial(
        embed_layer,
        mean_layer,
        dense_output_layer,
        log_softmax_layer
    )

    return model 

tmp_model = classifier()
print(type(tmp_model))


# training the model 
from trax.supervised import training 
batch_size = 16 
rnd.seed(271)
train_task = training.TrainTask(
        labeled_data=train_generator(batch_size=batch_size, shuffle=True),
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=trax.optimizers.Adam(0.01),
        n_steps_per_checkpoint=10)
eval_task = training.EvalTask(
        labeled_data=val_generator(batch_size=batch_size, shuffle=True),
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()])

model = classifier()

output_dir = '~/model/'
output_dir_expand = os.path.expanduser(output_dir)
print(output_dir_expand)

def train_model(classifier, train_task, eval_task, n_steps, output_dir):
    training_loop = training.Loop(classifier, train_task, eval_tasks=[eval_task], output_dir=output_dir)
    training_loop.run(n_steps=n_steps)
    return training_loop

training_loop = train_model(model, train_task, eval_task, 100, output_dir_expand)

tmp_train_generator = train_generator(16)
tmp_batch = next(tmp_train_generator)

tmp_inputs, tmp_targets, tmp_example_weights = tmp_batch
print(len(tmp_batch))
print(tmp_inputs.shape)
print(tmp_targets.shape)
print(tmp_example_weights.shape)

tmp_pred = training_loop.eval_model(tmp_inputs)
print(tmp_pred.shape)
tmp_pred

tmp_is_positive = tmp_pred[:, 1] > tmp_pred[:, 0]
for i, p in enumerate(tmp_is_positive):
    print(f"Neg log prob {tmp_pred[i, 0]:.4f} \t Pos log prob {tmp_pred[i, 1]:.4f} is positive? {p} \t actual {tmp_targets[i]}")

display(tmp_is_positive)
tmp_is_positive_int = tmp_is_positive.astype(np.int32)
display(tmp_is_positive_int)
tmp_is_positive_float = tmp_is_positive.astype(np.float32)
display(tmp_is_positive_float)

def compute_accuracy(preds, y, y_weights):
    is_pos = preds[:, 1] > preds[:, 0]
    is_pos_int = is_pos.astype(np.int32)
    correct = is_pos_int == y 
    sum_weights = np.sum(y_weights)
    correct_float = correct.astype(np.float32)
    weighted_correct_float = correct_float * y_weights
    weighted_num_correct = np.sum(weighted_correct_float)
    accuracy = weighted_num_correct / sum_weights 
    return accuracy, weighted_num_correct, sum_weights

tmp_val_generator = val_generator(64)
tmp_batch = next(tmp_val_generator)
tmp_inputs, tmp_targets, tmp_example_weights = tmp_batch
print(tmp_inputs.shape)
print(tmp_targets.shape)
print(tmp_example_weights.shape)
tmp_pred = training_loop.eval_model(tmp_inputs)
print(tmp_pred.shape)
tmp_acc, tmp_num_correct, tmp_num_predictions = compute_accuracy(preds=tmp_pred, y=tmp_targets, y_weights = tmp_example_weights)
print(100*tmp_acc)
print(tmp_num_correct)
print(tmp_num_predictions)

# testing your model on validation data 
# excercise 08

def test_model(generator, model):
    accuracy = 0.
    total_num_correct = 0
    total_num_pred = 0 

    for batch in generator:
        inputs = batch[0]
        targets = batch[1]
        example_weights = batch[2]

        pred = model(inputs)
        batch_accuracy, batch_num_correct, batch_num_pred = compute_accuracy(preds=pred, y=targets, y_weights=example_weights)
        total_num_correct += batch_num_correct
        total_num_pred += batch_num_pred

    accuracy = total_num_correct / total_num_pred
    return accuracy 


model = training_loop.eval_model
accuracy = test_model(test_generator(16), model)
print(accuracy)

#testing with your own input 
def predict(sentence):
    inputs = np.array(tweet_to_tensor(sentence, vocab_dict=Vocab))
    inputs = inputs[1, :]
    pred_probs = model(inputs)
    preds = int(pred_probs[0, 1] > pred_probs[0, 0])
    sentiment = 'negative'
    if preds == 1:
        sentiment = 'positive'
    return preds, sentiment

sentence = "It's such a nice day, think i'll be taking Sid to Ramsgate fish and chips for lunch at Peter's fish factory and then the beach maybe"
tmp_pred, tmp_sentiment = pedict


