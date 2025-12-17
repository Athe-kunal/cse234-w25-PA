import argparse
import json
from scipy.optimize import minimize_scalar


def load_model_config(model_config_path):
    """
    Utility function to read a JSON model config file.
    Args:
        model_config_path (str): Path to the model config JSON file.
    Returns:
        dict: Loaded model configuration as a dictionary.
    """
    with open(model_config_path, "r") as f:
        config = json.load(f)
    return config


def model_training_cost_analysis_llama(model_config_path):
    model_config = load_model_config(model_config_path=model_config_path)
    hidden_size = model_config["hidden_size"]
    intermediate_size = model_config["intermediate_size"]
    num_layers = model_config["num_hidden_layers"]
    vocab_size = model_config["vocab_size"]

    # Per-layer parameters
    params_per_layer = (
        4 * hidden_size * hidden_size  # W_Q, W_K, W_V, W_O
        + 3 * hidden_size * intermediate_size  # MLP (SwiGLU: gate, up, down)
        + 2 * hidden_size  # 2 RMSNorms per layer
    )

    # Total parameters
    total_params = (
        num_layers * params_per_layer  # All transformer layers
        + vocab_size * hidden_size  # Word embedding
        + hidden_size * vocab_size  # LM head (not tied)
        + hidden_size  # Final RMSNorm
    )
    flops_layer_TF = 2 * total_params / 1e12
    peak_memory_GB = 2 * total_params / 1e9
    return total_params, flops_layer_TF, peak_memory_GB


def model_training_cost_analysis_deepseek(model_config_path):
    model_config = load_model_config(model_config_path=model_config_path)
    hidden_size = model_config["hidden_size"]  # 7168
    num_layers = model_config["num_hidden_layers"]  # 61
    vocab_size = model_config["vocab_size"]  # 129280

    # MLA attention params (low-rank projections)
    q_lora_rank = model_config["q_lora_rank"]  # 1536
    kv_lora_rank = model_config["kv_lora_rank"]  # 512
    num_heads = model_config["num_attention_heads"]  # 128
    qk_head_dim = (
        model_config["qk_nope_head_dim"] + model_config["qk_rope_head_dim"]
    )  # 192
    v_head_dim = model_config["v_head_dim"]  # 128

    # MoE config
    n_routed = model_config["n_routed_experts"]  # 256
    n_shared = model_config["n_shared_experts"]  # 1
    num_experts_per_tok = model_config["num_experts_per_tok"]  # 8
    moe_intermediate = model_config["moe_intermediate_size"]  # 2048
    intermediate_size = model_config["intermediate_size"]  # 18432

    # Dense vs MoE layers
    num_dense_layers = model_config["first_k_dense_replace"]  # 3
    num_moe_layers = num_layers - num_dense_layers  # 58

    # MLA attention params per layer
    attn_params_per_layer = (
        hidden_size * q_lora_rank  # Q down-projection
        + q_lora_rank * (num_heads * qk_head_dim)  # Q up-projection
        + hidden_size * kv_lora_rank  # KV down-projection
        + kv_lora_rank * (num_heads * (qk_head_dim + v_head_dim))  # KV up-projection
        + (num_heads * v_head_dim) * hidden_size  # O projection
    )

    # Dense MLP params (for first 3 layers) - SwiGLU has 3 matrices
    dense_mlp_params_per_layer = 3 * hidden_size * intermediate_size

    # MoE MLP params per layer (total params, all experts)
    moe_mlp_params_per_layer = (
        3 * (n_routed + n_shared) * moe_intermediate * hidden_size
    )

    # MoE MLP activated params per layer (only activated experts)
    activated_moe_mlp_params_per_layer = (
        3 * (num_experts_per_tok + n_shared) * moe_intermediate * hidden_size
    )

    # === Total parameters (all weights) ===
    total_params = (
        # Attention (all layers)
        num_layers * attn_params_per_layer
        # RMSNorm (2 per layer + 1 final)
        + num_layers * 2 * hidden_size
        + hidden_size
        # Dense MLP layers (first 3)
        + num_dense_layers * dense_mlp_params_per_layer
        # MoE MLP layers (remaining 58)
        + num_moe_layers * moe_mlp_params_per_layer
        # Embeddings
        + 2 * vocab_size * hidden_size  # embed + LM head
    )

    # === Activated parameters per forward pass ===
    # For FLOPs, we care about what's actually computed
    activated_params_per_layer = (
        attn_params_per_layer + 2 * hidden_size  # RMSNorm per layer
    )
    # Dense layers use full MLP, MoE layers use only activated experts
    activated_total_params = (
        num_layers * activated_params_per_layer
        + num_dense_layers * dense_mlp_params_per_layer
        + num_moe_layers * activated_moe_mlp_params_per_layer
        + 2 * vocab_size * hidden_size  # embed + LM head
        + hidden_size  # Final RMSNorm
    )

    flops_layer_TF = 2 * activated_total_params / 1e12
    peak_memory_GB = 2 * total_params / 1e9
    return total_params, flops_layer_TF, peak_memory_GB


def get_optimal_N_D_from_cost(cost_budget):
    """
    cost_budget:  a monetary training budget (in dollars)
    Returns:
        N: Optimal total model parameters (in absolute numbers)
        D: Optimal number of training tokens (in absolute numbers)
        training_budget_flops: Effective total training FLOPs (in FLOPs)
        best_gpu: name of the selected GPU (one of 'A100', 'V100', 'T4')
    """
    gpus = {"A100": (4, 312), "V100": (2.5, 125), "T4": (1.0, 65)}
    MFU = 0.4
    best_gpu = None
    best_flops_per_dollar = 0

    for gpu_name, (cost_per_hour, peak_tflops) in gpus.items():
        effective_flops_per_hour = peak_tflops * MFU
        flops_per_dollar = effective_flops_per_hour / cost_per_hour
        if flops_per_dollar > best_flops_per_dollar:
            best_flops_per_dollar = flops_per_dollar
            best_gpu = gpu_name

    training_budget_flops = best_flops_per_dollar * 1e12 * cost_budget

    C = training_budget_flops

    def loss_given_N(N):
        D = C / (6 * N)
        return 406.4 / (N**0.34) + 410.7 / (D**0.29) + 1.69

    # N must be positive and D must be positive
    # D = C/(6N) > 0 is always true for N > 0
    # Search over reasonable range of N (e.g., 1e6 to 1e15)
    result = minimize_scalar(loss_given_N, bounds=(1e6, 1e15), method="bounded")

    N = result.x
    D = C / (6 * N)

    return int(N), int(D), training_budget_flops, best_gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training cost analysis")
    parser.add_argument("--model_config", type=str, help="Path to model config file")
    parser.add_argument(
        "--training_budget", type=float, default=None, help="Training budget"
    )
    args = parser.parse_args()

    if args.model_config:
        if "deepseek" in args.model_config:
            num_parameters, num_flops, memory_cost = (
                model_training_cost_analysis_deepseek(args.model_config)
            )
        elif "llama" in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_llama(
                args.model_config
            )
        else:
            raise ValueError("Unknown LLM Type!")
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(
            args.training_budget
        )
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")
