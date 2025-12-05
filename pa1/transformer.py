import functools
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import auto_diff as ad
import torch
from torchvision import datasets, transforms

max_len = 28


def feedforward(X: ad.Node, weight: ad.Node, bias: ad.Node) -> ad.Node:
    return ad.matmul(X, weight) + bias


def single_head_attention(
    X: ad.Node, WQ: ad.Node, WK: ad.Node, WV: ad.Node, Wo: ad.Node, model_dim: int
) -> ad.Node:
    Q = ad.matmul(X, WQ)
    K = ad.transpose(ad.matmul(X, WK), -1, -2)
    QK = ad.softmax(ad.div_by_const(ad.matmul(Q, K), model_dim**0.5))
    attn = ad.matmul(QK, ad.matmul(X, WV))
    attn_out = ad.matmul(attn, Wo)
    return attn_out


def transformer(
    X: ad.Node,
    nodes: List[ad.Node],
    model_dim: int,
    seq_length: int,
    eps,
    batch_size,
    num_classes,
) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, model_dim), denoting the input data.
    nodes: List[ad.Node]
        Nodes you would need to initialize the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """
    # X: (batch_size, seq_length, model_dim)
    # nodes order:
    W_Q_val, W_K_val, W_V_val, W_1_val, b_1_val, W_2_val, b_2_val, W_O_val = nodes
    attn = single_head_attention(
        X, W_Q_val, W_K_val, W_V_val, W_O_val, model_dim
    )  # (batch_size, seq_length, model_dim)

    # Residual connection (add input to attn output)
    attn_out = attn + X

    # LayerNorm after attention
    norm1 = ad.layernorm(attn_out, normalized_shape=[model_dim], eps=eps)  # last dim

    ff1 = feedforward(norm1, W_1_val, b_1_val)  # (batch_size, seq_length, model_dim)
    ff1_act = ad.relu(ff1)
    logits = feedforward(
        ff1_act, W_2_val, b_2_val
    )  # (batch_size, seq_length, num_classes)

    return logits


def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.

    As we need to compute the average loss
    """
    # Z: (batch_size, num_classes), y_one_hot: (batch_size, num_classes)
    # Compute softmax probabilities
    probs = ad.softmax(Z, dim=1)  # (batch_size, num_classes)
    # Compute negative log likelihood for correct class
    log_probs = ad.log(probs)  # (batch_size, num_classes)
    loss_per_example = ad.mul_by_const(
        ad.mul(y_one_hot, log_probs), -1
    )  # (batch_size, num_classes)
    loss_sum = ad.sum_op(loss_per_example, dim=(1,), keepdim=False)  # (batch_size,)
    mean_loss = ad.mean(loss_sum, dim=(0,), keepdim=False)  # scalar ()
    return mean_loss


def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> tuple[List[torch.Tensor], float]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    num_examples = X.shape[0]
    num_batches = (
        num_examples + batch_size - 1
    ) // batch_size  # Compute the number of batches
    total_loss = 0.0

    for i in range(num_batches):
        # Get the mini-batch data
        start_idx = i * batch_size
        if start_idx + batch_size > num_examples:
            continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]

        # Compute forward and backward passes
        # TODO: Your code here
        result = f_run_model(X_batch, y_batch, model_weights)
        (
            y_pred,
            loss,
            grad_W_Q,
            grad_W_K,
            grad_W_V,
            grad_W_1,
            grad_b_1,
            grad_W_2,
            grad_b_2,
            grad_W_O,
        ) = result

        # Update each weight individually
        # Hint: You can update the tensor using something like below:
        # W_Q -= lr * grad_W_Q.sum(dim=0)

        W_Q, W_K, W_V, W_1, b_1, W_2, b_2, W_O = model_weights
        W_Q = W_Q - lr * grad_W_Q.sum(dim=0)
        W_K = W_K - lr * grad_W_K.sum(dim=0)
        W_V = W_V - lr * grad_W_V.sum(dim=0)
        W_O = W_O - lr * grad_W_O.sum(dim=0)
        W_1 = W_1 - lr * grad_W_1.sum(dim=0)
        b_1 = b_1 - lr * grad_b_1.sum(dim=0)
        W_2 = W_2 - lr * grad_W_2.sum(dim=0)
        b_2 = b_2 - lr * grad_b_2.sum(dim=0)

        model_weights = [W_Q, W_K, W_V, W_1, b_1, W_2, b_2, W_O]

        # Accumulate the loss
        total_loss += loss.item() if isinstance(loss, torch.Tensor) else float(loss)
    # Compute the average loss
    average_loss = total_loss / num_batches
    print("Avg_loss:", average_loss)

    # TODO: Your code here
    # You should return the list of parameters and the loss
    return model_weights, average_loss


def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params

    # TODO: Tune your hyperparameters here
    # Hyperparameters
    input_dim = 28  # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10  #
    model_dim = 128  #
    eps = 1e-5

    # - Set up the training settings.
    num_epochs = 20
    batch_size = 50
    lr = 0.02

    # Create Variable nodes for inputs and weights
    X = ad.Variable(name="X")
    y_groundtruth = ad.Variable(name="y")
    W_Q = ad.Variable(name="WQ")
    W_K = ad.Variable(name="WK")
    W_V = ad.Variable(name="WV")
    W_O = ad.Variable(name="Wo")
    W_1 = ad.Variable(name="W1")
    b_1 = ad.Variable(name="b1")
    W_2 = ad.Variable(name="W2")
    b_2 = ad.Variable(name="b2")

    # Build the transformer graph
    model_weights_nodes = [W_Q, W_K, W_V, W_1, b_1, W_2, b_2, W_O]
    y_predict = transformer(
        X, model_weights_nodes, model_dim, seq_length, eps, batch_size, num_classes
    )

    # Average over sequence length for classification
    # y_predict is (batch_size, seq_length, num_classes), need to average to (batch_size, num_classes)
    y_predict_avg = ad.mean(
        y_predict, dim=(1,), keepdim=False
    )  # (batch_size, num_classes)

    # Build loss graph
    loss: ad.Node = softmax_loss(y_predict_avg, y_groundtruth, batch_size)

    # Construct the backward graph
    grads: List[ad.Node] = ad.gradients(loss, model_weights_nodes)
    evaluator = ad.Evaluator([y_predict_avg, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict_avg])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    # Convert the train dataset to NumPy arrays
    X_train = (
        train_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    )  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = (
        test_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    )  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(
        sparse_output=False
    )  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    # Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))

    def f_run_model(
        X_batch: torch.Tensor, y_batch: torch.Tensor, model_weights: List[torch.Tensor]
    ):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        W_Q_val, W_K_val, W_V_val, W_1_val, b_1_val, W_2_val, b_2_val, W_O_val = (
            model_weights
        )
        result = evaluator.run(
            input_values={
                X: X_batch,
                y_groundtruth: y_batch,
                W_Q: W_Q_val,
                W_K: W_K_val,
                W_V: W_V_val,
                W_O: W_O_val,
                W_1: W_1_val,
                b_1: b_1_val,
                W_2: W_2_val,
                b_2: b_2_val,
            }
        )
        return result

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (
            num_examples + batch_size - 1
        ) // batch_size  # Compute the number of batches
        total_loss = 0.0
        all_logits = []
        for i in range(num_batches):
            # Get the mini-batch data
            start_idx = i * batch_size
            if start_idx + batch_size > num_examples:
                continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            W_Q_val, W_K_val, W_V_val, W_1_val, b_1_val, W_2_val, b_2_val, W_O_val = (
                model_weights
            )
            logits = test_evaluator.run(
                {
                    X: X_batch,
                    W_Q: W_Q_val,
                    W_K: W_K_val,
                    W_V: W_V_val,
                    W_O: W_O_val,
                    W_1: W_1_val,
                    b_1: b_1_val,
                    W_2: W_2_val,
                    b_2: b_2_val,
                }
            )
            all_logits.append(logits[0])
        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Train the model.
    X_train, X_test, y_train, y_test = (
        torch.tensor(X_train),
        torch.tensor(X_test),
        torch.DoubleTensor(y_train),
        torch.DoubleTensor(y_test),
    )
    model_weights: List[torch.Tensor] = [
        W_Q_val,
        W_K_val,
        W_V_val,
        W_1_val,
        b_1_val,
        W_2_val,
        b_2_val,
        W_O_val,
    ]
    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        # Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label== y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
