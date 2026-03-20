"""Small runnable examples for the Tensor class.

Run:
    python3 tensor_walkthrough.py
"""

import numpy as np

from gpt_peft_np import Tensor, cross_entropy, softmax


def print_tensor(name, tensor):
    print(f"{name}:")
    print(f"  data = {tensor.data}")
    print(f"  grad = {tensor.grad}")
    print()


def example_1_basic_math():
    print("Example 1: basic Tensor math and gradients")
    print("We create three learnable values: x, y, and b.")
    print("Then we compute z = (x * y) + b and run backward().")
    print()

    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    b = Tensor(1.0, requires_grad=True)

    z = (x * y) + b

    print_tensor("x", x)
    print_tensor("y", y)
    print_tensor("b", b)
    print_tensor("z = (x * y) + b", z)

    print("Calling z.backward() starts backpropagation from the final result.")
    print("Because z is a single number, the starting gradient is 1.0.")
    print()

    z.backward()

    print("After backward:")
    print("dz/dx = y, so x.grad should be 3")
    print("dz/dy = x, so y.grad should be 2")
    print("dz/db = 1, so b.grad should be 1")
    print()

    print_tensor("x", x)
    print_tensor("y", y)
    print_tensor("b", b)


def example_2_broadcasting_manual():
    print("Example 2: broadcasting")
    print("Here a bias vector is added across two rows.")
    print("We manually create a scalar loss by indexing and adding all elements.")
    print()

    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    bias = Tensor([10.0, 20.0], requires_grad=True)

    out = x + bias
    loss = out[0, 0] + out[0, 1] + out[1, 0] + out[1, 1]

    print_tensor("x", x)
    print_tensor("bias", bias)
    print_tensor("out = x + bias", out)
    print_tensor("loss = sum(out)", loss)

    loss.backward()

    print("Each output element contributes 1 to the loss.")
    print("So x.grad becomes all ones.")
    print("The bias was reused across 2 rows, so bias.grad becomes [2, 2].")
    print()

    print_tensor("x", x)
    print_tensor("bias", bias)


def example_3_matrix_multiply():
    print("Example 3: matrix multiplication")
    print("This is the core operation behind linear layers in neural networks.")
    print()

    x = Tensor([[1.0, 2.0]], requires_grad=True)
    w = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    b = Tensor([0.5, -0.5], requires_grad=True)

    out = (x @ w) + b
    loss = out[0, 0] + out[0, 1]

    print_tensor("x", x)
    print_tensor("w", w)
    print_tensor("b", b)
    print_tensor("out = (x @ w) + b", out)

    loss.backward()

    print("After backward, gradients tell us how much each input affected the final scalar.")
    print_tensor("x", x)
    print_tensor("w", w)
    print_tensor("b", b)


def example_4_shape_ops():
    print("Example 4: reshape, transpose, and slicing")
    print("These do not change the meaning of the data, only how we view or select it.")
    print()

    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    reshaped = x.reshape(3, 2)
    transposed = x.transpose(1, 0)
    picked = transposed[0, 1] + transposed[2, 1]

    print_tensor("x", x)
    print_tensor("reshaped = x.reshape(3, 2)", reshaped)
    print_tensor("transposed = x.transpose(1, 0)", transposed)
    print_tensor("picked = transposed[0, 1] + transposed[2, 1]", picked)

    picked.backward()

    print("Only the selected positions receive gradient.")
    print("transposed[0, 1] maps to x[1, 0]")
    print("transposed[2, 1] maps to x[1, 2]")
    print()

    print_tensor("x", x)


def example_5_softmax_and_loss():
    print("Example 5: softmax and cross-entropy")
    print("This is the kind of setup used for classification or next-token prediction.")
    print()

    logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
    probs = softmax(logits, axis=-1)
    targets = np.array([0], dtype=np.int64)
    loss = cross_entropy(logits, targets)

    print_tensor("logits", logits)
    print_tensor("probs = softmax(logits)", probs)
    print_tensor("loss = cross_entropy(logits, targets=[0])", loss)

    loss.backward()

    print("The model predicted class 0 most strongly, so the loss is fairly small.")
    print("The gradients show how each logit should move to reduce the loss.")
    print()

    print_tensor("logits", logits)


def main():
    print("Tensor walkthrough")
    print("==================")
    print()
    print("This script demonstrates how the homemade Tensor class stores data,")
    print("builds a computation graph, and computes gradients with backward().")
    print()

    example_1_basic_math()
    print("-" * 60)
    example_2_broadcasting_manual()
    print("-" * 60)
    example_3_matrix_multiply()
    print("-" * 60)
    example_4_shape_ops()
    print("-" * 60)
    example_5_softmax_and_loss()


if __name__ == "__main__":
    main()
