from typing import Any, Dict, List
import torch
from auto_diff import *


class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, node_A: Node, node_B: Node, normalized_shape: List[int], eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={"normalized_shape": normalized_shape, "eps": eps},
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        # First perform matrix multiplication using matmul's compute
        matmul_node = Node(
            inputs=[node.inputs[0], node.inputs[1]], op=matmul, name="temp_matmul"
        )
        matmul_result = matmul.compute(matmul_node, input_values)

        # Then apply layer normalization using layernorm's compute
        layernorm_node = Node(
            inputs=[node.inputs[0]],  # This will be replaced by matmul_result
            op=layernorm,
            attrs={"normalized_shape": node.normalized_shape, "eps": node.eps},
            name="temp_layernorm",
        )
        y = layernorm.compute(layernorm_node, [matmul_result])
        return y

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        node_a, node_b = node.inputs[0], node.inputs[1]

        # First compute matmul result as a node
        matmul_result = matmul(node_a, node_b)

        # Compute gradient through layernorm to get gradient of matmul result
        layernorm_node = Node(
            inputs=[matmul_result],
            op=layernorm,
            attrs={"normalized_shape": node.normalized_shape, "eps": node.eps},
            name=f"LayerNorm({matmul_result.name})",
        )
        layernorm_grad = layernorm.gradient(layernorm_node, output_grad)[0]

        # Then compute gradient through matmul to get gradients for A and B
        matmul_node = Node(
            inputs=[node_a, node_b],
            op=matmul,
            name=f"MatMul({node_a.name}@{node_b.name})",
        )
        matmul_grads = matmul.gradient(matmul_node, layernorm_grad)

        return matmul_grads


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(self, node_A: Node, node_B: Node, dim: int = -1) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={"dim": dim},
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        # First perform matrix multiplication using matmul's compute
        matmul_node = Node(
            inputs=[node.inputs[0], node.inputs[1]], op=matmul, name="temp_matmul"
        )
        matmul_result = matmul.compute(matmul_node, input_values)

        # Then apply softmax using softmax's compute
        softmax_node = Node(
            inputs=[node.inputs[0]],  # This will be replaced by matmul_result
            op=softmax,
            attrs={"dim": node.dim},
            name="temp_softmax",
        )
        y = softmax.compute(softmax_node, [matmul_result])
        return y

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        node_a, node_b = node.inputs[0], node.inputs[1]

        # First compute matmul result as a node
        matmul_result = matmul(node_a, node_b)

        # Compute softmax output (needed for softmax gradient)
        softmax_y = softmax(matmul_result, dim=node.dim)

        # Compute gradient through softmax to get gradient of matmul result
        # Softmax gradient uses the softmax output node directly
        matmul_grad = softmax.gradient(softmax_y, output_grad)[0]

        # Then compute gradient through matmul to get gradients for A and B
        matmul_node = Node(
            inputs=[node_a, node_b],
            op=matmul,
            name=f"MatMul({node_a.name}@{node_b.name})",
        )
        matmul_grads = matmul.gradient(matmul_node, matmul_grad)

        return matmul_grads


# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()
