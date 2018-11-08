"""
Test the ESIM model on some preprocessed dataset.
"""
# Aurelien Coet, 2018.

import os
import time
import pickle
import argparse
import torch
import json

from sklearn import metrics
from torch.utils.data import DataLoader
from esim.dataset import NLIDataset
from esim.model import ESIM
from esim.utils import correct_predictions


def test(model, dataloader):
    """
    Test the accuracy of a model on some dataset.

    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.

    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device

    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0

    pred = []
    true = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            batch_start = time.time()

            # Move input and output data to the GPU if one is used.
            premises = batch['premise'].to(device)
            premises_lengths = batch['premise_length'].to(device)
            hypotheses = batch['hypothesis'].to(device)
            hypotheses_lengths = batch['hypothesis_length'].to(device)
            labels = batch['label'].to(device)

            _, probs = model(premises,
                             premises_lengths,
                             hypotheses,
                             hypotheses_lengths)

            accuracy += correct_predictions(probs, labels)
            batch_time += time.time() - batch_start

            _, out_classes = probs.max(dim=1)
            pred.extend(out_classes.cpu().tolist())
            true.extend(labels.cpu().tolist())

    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))
    print('=== confusion matrix ===')
    print(metrics.confusion_matrix(true, pred))
    print(metrics.f1_score(true, pred))
    pickle.dump([pred, true], open('tmp.result', 'wb'))
    return batch_time, total_time, accuracy


def main(test_file,
         pretrained_file,
         batch_size=32):
    """
    Test the ESIM model with pretrained weights on some dataset.

    Args:
        test_file: The path to a file containing preprocessed NLI data.
        pretrained_file: The path to a checkpoint produced by the
            'train_model' script.
        batch_size: The size of the batches used for testing. Defaults to 32.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for testing ", 20 * "=")

    print("\t* Loading test data...")
    with open(test_file, 'rb') as pkl:
        test_data = NLIDataset(pickle.load(pkl), max_premise_length=100, max_hypothesis_length=12)
    print('\t* Number of data: %s' % len(test_data))
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    print("\t* Building model...")
    # hacking the model
    checkpoint = torch.load(pretrained_file)
    vocab_size, embedding_dim = checkpoint['model']['_word_embedding.weight'].size()
    num_classes = 2
    hidden_size = 300

    model = ESIM(vocab_size,
                 embedding_dim,
                 hidden_size,
                 num_classes=num_classes,
                 device=device).to(device)
    if 'fixed_embedding' in checkpoint['model']:
        checkpoint['model'].pop('fixed_embedding')
    model.load_state_dict(checkpoint['model'])

    print(20 * "=",
          " Testing ESIM model on device: {} ".format(device),
          20 * "=")
    batch_time, total_time, accuracy = test(model, test_loader)

    print("-> Average batch processing time: {:.4f}s, total test time:\
 {:.4f}s, accuracy: {:.4f}%".format(batch_time, total_time, (accuracy*100)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the ESIM model on\
 some dataset')
    parser.add_argument('--checkpoint',
                        help="Path to a checkpoint with a pretrained model")
    parser.add_argument('--batch-size', default=32,
                        help='Batch size for test')
    parser.add_argument('--test-data', default='../config/test.json',
                        help='Path to a test file')
    args = parser.parse_args()

    main(os.path.normpath(args.test_data),
         args.checkpoint,
         batch_size=args.batch_size)
