import numpy as np

from mpiwrapper import mpi
from rng import get_rng, rng_context


class Linear:
    """Simple linear layer y = xW + b"""

    def __init__(self, in_features, out_features):
        # Use default RNG for all other operations - no need for context
        self.weight = get_rng().randn(in_features, out_features) * 0.01
        self.bias = np.zeros(out_features)

    def __call__(self, x):
        return np.dot(x, self.weight) + self.bias


class Expert:
    """Expert network with one hidden layer and ReLU activation"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        # Use rank-specific RNG for expert initialization
        with rng_context("expert"):
            self.fc1 = Linear(input_dim, hidden_dim)
            self.fc2 = Linear(hidden_dim, output_dim)

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)  # ReLU
        return self.fc2(hidden)


class Router:
    """Routes inputs to experts using softmax-based gating"""

    def __init__(self, input_dim, num_experts):
        # Router should be consistent across all ranks, so use default RNG
        self.linear = Linear(input_dim, num_experts)

    def __call__(self, x, topk=1):
        logits = self.linear(x)

        # Softmax for routing probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Select top-k experts
        indices = np.argsort(-probs, axis=1)[:, :topk]
        gates = np.take_along_axis(probs, indices, axis=1)

        # Normalize gates to sum to 1
        gates = gates / np.sum(gates, axis=1, keepdims=True)

        return indices, gates


class ColumnParallelLinear:
    """
    Column-parallel linear layer: weights sharded along OUTPUT dimension.
    No communication in forward pass - output stays partitioned.

    Use as first layer in MLP (feeds into RowParallelLinear).
    """

    def __init__(self, in_features, out_features):
        self.rank = mpi.get_rank()
        self.world_size = mpi.get_size()

        assert (
            out_features % self.world_size == 0
        ), f"Output features ({out_features}) must be divisible by world size ({self.world_size})"

        self.in_features = in_features
        self.out_features_global = out_features
        self.local_out_features = out_features // self.world_size

        # Weight: (in_features, local_out_features) - sharded along columns
        self.weight = get_rng().randn(in_features, self.local_out_features) * 0.01
        self.bias = get_rng().randn(self.local_out_features)

    def __call__(self, x):
        if x.shape[0] == 0:
            return np.zeros((0, self.local_out_features))

        # Local matmul - NO communication, output stays partial
        # x: (batch, in_features), weight: (in_features, local_out_features)
        return x @ self.weight + self.bias  # (batch, local_out_features)


class RowParallelLinear:
    """
    Row-parallel linear layer: weights sharded along INPUT dimension.
    Requires all-reduce in forward pass to combine partial results.

    Use as second layer in MLP (receives partitioned input from ColumnParallelLinear).
    """

    def __init__(self, in_features, out_features):
        self.rank = mpi.get_rank()
        self.world_size = mpi.get_size()

        assert (
            in_features % self.world_size == 0
        ), f"Input features ({in_features}) must be divisible by world size ({self.world_size})"

        self.in_features_global = in_features
        self.local_in_features = in_features // self.world_size
        self.out_features = out_features

        # Weight: (local_in_features, out_features) - sharded along rows
        self.weight = get_rng().randn(self.local_in_features, out_features) * 0.01
        self.bias = get_rng().randn(out_features)

    def __call__(self, x):
        if x.shape[0] == 0:
            return np.zeros((0, self.out_features))

        # x is already partitioned: (batch, local_in_features)
        # Local matmul gives partial contribution to output
        local_out = x @ self.weight  # (batch, out_features) - partial

        # All-reduce to sum partial results from all ranks
        result = mpi.allreduce(local_out)

        # Add bias after all-reduce
        return result + self.bias


# Alias for backwards compatibility - uses simple column-parallel with allgather
class ShardedLinear:
    """
    Simple sharded linear with allgather (column-parallel style).
    Each rank computes partial output, then allgather combines them.
    """

    def __init__(self, in_features, out_features):
        self.rank = mpi.get_rank()
        self.world_size = mpi.get_size()

        assert (
            out_features % self.world_size == 0
        ), f"Output features ({out_features}) must be divisible by world size ({self.world_size})"

        self.in_features = in_features
        self.out_features_global = out_features
        self.local_out_features = out_features // self.world_size

        self.weight = get_rng().randn(in_features, self.local_out_features) * 0.01
        self.bias = get_rng().randn(self.local_out_features)

    def __call__(self, x):
        if x.shape[0] == 0:
            return np.zeros((0, self.out_features_global))

        # Local matmul
        local_out = x @ self.weight + self.bias  # (batch, local_out_features)

        # Allgather to combine partial outputs
        gathered = mpi.allgather(local_out)
        mpi.barrier()
        return np.concatenate(gathered, axis=1)  # (batch, out_features_global)


class ShardedExpert:
    """Expert network with one hidden layer and ReLU activation, sharded across processes

    Uses Megatron-style tensor parallelism:
    - fc1: ColumnParallelLinear (no comm, output partitioned)
    - fc2: RowParallelLinear (all-reduce to combine)

    This requires only 1 all-reduce per expert forward pass.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        # Use rank-specific RNG for expert initialization
        with rng_context("expert"):
            self.fc1 = ShardedLinear(input_dim, hidden_dim)
            self.fc2 = ShardedLinear(hidden_dim, output_dim)

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)  # ReLU
        return self.fc2(hidden)


class MoE_TP:
    """
    Distributed Mixture of Experts using MPI for tensor parallelism

    TP-style MoE:
    - Each process holds a portion of every expert (sharded experts)
    - Router is replicated on all processes
    - All-to-all and all-gather communication patterns for processing

    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for each expert
        output_dim (int): Output dimension
        num_experts (int): Total number of experts in the model
        topk (int): Number of experts to route each input to
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)
        self.rank = mpi.get_rank()
        self.world_size = mpi.get_size()

        # Create router (replicated on all processes)
        with rng_context("router"):
            self.router = Router(input_dim, num_experts)

        # Create sharded experts - each expert is sharded across all processes
        with rng_context("expert"):
            self.experts = [
                ShardedExpert(input_dim, hidden_dim, output_dim)
                for _ in range(num_experts)
            ]

        print(f"[Rank {self.rank}] Holding portions of all {num_experts} experts")

    def forward(self, x):
        """
        Distributed forward pass through the MoE model using tensor parallelism
        with optimized batch processing

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Initialize output tensor
        outputs = np.zeros((batch_size, self.output_dim))

        # 1. Compute the routing indices and gates for each input
        # Router is replicated, so all ranks get same routing decisions
        indices, gates = self.router(x, self.topk)

        # 2. Process experts with TP style
        # Each ShardedExpert handles its own communication internally
        for k in range(self.topk):
            for i in range(batch_size):
                expert_idx = indices[i, k]
                gate = gates[i, k]
                item = x[i : i + 1]  # (1, input_dim)
                # ShardedExpert internally does: ColumnParallel -> ReLU -> RowParallel (all-reduce)
                expert_output = self.experts[expert_idx](item)
                outputs[i] += gate * expert_output[0]

        return outputs

    def __call__(self, x):
        return self.forward(x)


class SimpleMoE:
    """
    Simple reference implementation of Mixture of Experts.

    This class implements a basic MoE model that routes inputs to a subset
    of experts and combines their outputs using learned gating weights.

    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for each expert
        output_dim (int): Output dimension
        num_experts (int): Number of expert networks
        topk (int): Number of experts to route each input to
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)

        # Create router network
        with rng_context("router"):
            self.router = Router(input_dim, num_experts)

        # Create expert networks
        with rng_context("expert"):
            self.experts = [
                Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
            ]

    def forward(self, x):
        """
        Forward pass through the MoE model

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Get expert assignments and gates
        indices, gates = self.router(x, self.topk)

        # Initialize output tensor
        outputs = np.zeros((batch_size, self.output_dim))

        # Compute weighted combination of expert outputs
        for k in range(self.topk):
            for i in range(batch_size):
                expert_idx = indices[i, k]
                gate = gates[i, k]
                item = x[i : i + 1]  # (1, input_dim)
                expert_output = self.experts[expert_idx](item)
                outputs[i] += gate * expert_output[0]

        return outputs

    def __call__(self, x):
        return self.forward(x)


class MoE_EP:
    """
    Distributed Mixture of Experts using MPI for expert parallelism

    EP-style MoE:
    Each process hosts exactly one expert. Router is replicated on all processes.

    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for each expert
        output_dim (int): Output dimension
        topk (int): Number of experts to route each input to
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts  # Total number of processes = number of experts
        self.topk = min(topk, self.num_experts)
        self.rank = mpi.get_rank()

        # Create router (replicated on all processes)
        with rng_context("router"):
            self.router = Router(input_dim, self.num_experts)

        # Create only one expert per process
        with rng_context("expert_with_rank"):
            self.expert = Expert(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        """
        Distributed forward pass through the MoE model

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Initialize output tensor
        outputs = np.zeros((batch_size, self.output_dim))

        # 1. Compute the routing indices and gates for each input
        # Router is replicated, so all ranks get the same routing decisions
        indices, gates = self.router(x, self.topk)

        # 2. Prepare data to send to each expert (rank)
        # For each rank, collect (input_data, batch_idx, k_idx) that need that expert
        send_to_rank = [[] for _ in range(self.num_experts)]

        for i in range(batch_size):
            for k in range(self.topk):
                expert_idx = indices[i, k]
                # Pack: input vector, batch index, k index, gate value
                send_to_rank[expert_idx].append((x[i], i, k, gates[i, k]))

        # 3. All-to-all: send inputs to the rank that owns each expert
        # Convert to list format for alltoall
        send_data = []
        for rank_id in range(self.num_experts):
            if len(send_to_rank[rank_id]) > 0:
                inputs = np.array([item[0] for item in send_to_rank[rank_id]])
                batch_indices = [item[1] for item in send_to_rank[rank_id]]
                k_indices = [item[2] for item in send_to_rank[rank_id]]
                gate_values = [item[3] for item in send_to_rank[rank_id]]
                send_data.append((inputs, batch_indices, k_indices, gate_values))
            else:
                # Empty array with correct shape for this rank
                send_data.append((np.zeros((0, self.input_dim)), [], [], []))

        # All-to-all exchange
        recv_data = mpi.alltoall(send_data)

        # 4. Process received inputs with local expert
        # recv_data[i] contains data from rank i that needs this rank's expert
        local_results = []
        for src_rank in range(self.num_experts):
            inputs, batch_indices, k_indices, gate_values = recv_data[src_rank]
            if len(batch_indices) > 0:
                # Process with local expert
                expert_outputs = self.expert(inputs)  # (num_inputs, output_dim)
                local_results.append(
                    (expert_outputs, batch_indices, k_indices, gate_values, src_rank)
                )
            else:
                local_results.append(
                    (np.zeros((0, self.output_dim)), [], [], [], src_rank)
                )

        # 5. Prepare results to send back to original ranks
        send_back = [None for _ in range(self.num_experts)]
        for (
            expert_outputs,
            batch_indices,
            k_indices,
            gate_values,
            src_rank,
        ) in local_results:
            send_back[src_rank] = (
                expert_outputs,
                batch_indices,
                k_indices,
                gate_values,
            )

        # All-to-all: send results back
        recv_results = mpi.alltoall(send_back)

        # 6. Aggregate results
        for src_rank in range(self.num_experts):
            expert_outputs, batch_indices, k_indices, gate_values = recv_results[
                src_rank
            ]
            for j, (batch_idx, k_idx, gate) in enumerate(
                zip(batch_indices, k_indices, gate_values)
            ):
                outputs[batch_idx] += gate * expert_outputs[j]

        return outputs

    def __call__(self, x):
        return self.forward(x)
