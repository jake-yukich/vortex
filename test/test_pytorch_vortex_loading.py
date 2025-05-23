import torch
import pytest
import os

# Add the root directory to sys.path to allow importing load_pytorch_vortex
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from load_pytorch_vortex import load_model
from vortex.model_pytorch.model import StripedHyena

# Define the configuration path (assuming it's relative to the root of the repo)
CONFIG_PATH_EVO2_1B_8K = "configs/evo2-1b-8k.yml" 

@pytest.fixture
def evo2_1b_8k_config_path():
    # Pytest fixture to provide the config path.
    # This also helps in checking if the config file exists before running tests.
    path = os.path.join(os.path.dirname(__file__), "..", CONFIG_PATH_EVO2_1B_8K)
    if not os.path.exists(path):
        pytest.skip(f"Config file not found: {path}")
    return path

def test_instantiate_evo2_1b_8k_config(evo2_1b_8k_config_path):
    """
    Tests model instantiation from the evo2-1b-8k.yml config.
    """
    model = load_model(config_path=evo2_1b_8k_config_path, device="cpu", verbose=False)
    assert isinstance(model, StripedHyena), "Model is not an instance of StripedHyena."
    
    # Check if the model is on CPU
    # For top-level parameters:
    assert all(p.device.type == 'cpu' for p in model.parameters()), "Not all model parameters are on CPU."
    # Specifically check the device attribute set in the model config
    assert model.config.device == 'cpu', "Model's internal config device is not set to CPU."
    # Check a buffer from an embedding layer as an example
    assert model.embedding_layer.word_embeddings.weight.device.type == 'cpu', "Embedding layer is not on CPU."


def test_forward_pass_evo2_1b_8k(evo2_1b_8k_config_path):
    """
    Tests a simple forward pass with the model loaded from evo2-1b-8k.yml config.
    """
    model = load_model(config_path=evo2_1b_8k_config_path, device="cpu", verbose=False)
    
    # Determine vocab_size from the loaded model's config
    vocab_size = model.config.get('vocab_size', 512) # Default if not specified
    
    sample_input = torch.randint(0, vocab_size, (1, 100), dtype=torch.long).to("cpu")
    
    with torch.no_grad():
        # Assuming model.forward returns (output, inference_params_dict)
        output, _ = model.forward(sample_input) 
    
    assert isinstance(output, torch.Tensor), "Output is not a torch.Tensor."
    
    # Assert output shape: (batch_size, sequence_length, vocab_size)
    # The last dimension of the output should be vocab_size for language models.
    expected_shape = (1, 100, vocab_size)
    assert output.shape == expected_shape, f"Output shape mismatch. Expected {expected_shape}, got {output.shape}."
    
    # Assert output dtype - this might depend on how the model is configured internally
    # For a CPU model without specific dtype changes, it often defaults to float32.
    # If the model internally casts to bfloat16 or float16, this needs adjustment.
    # Given the refactoring to PyTorch-native, it should be float32 unless specific layers cast.
    # Let's check the dtype of the unembedding layer's weight or a dense layer's output if possible.
    # The output of the unembedding layer determines the final output dtype.
    expected_dtype = model.unembed.word_embeddings.weight.dtype if hasattr(model.unembed, 'word_embeddings') else torch.float32
    assert output.dtype == expected_dtype, f"Output dtype mismatch. Expected {expected_dtype}, got {output.dtype}."


@pytest.mark.skipif(not os.environ.get("EVO2_1B_8K_WEIGHTS_PATH"), reason="EVO2_1B_8K_WEIGHTS_PATH not set")
def test_load_weights_evo2_1b_8k(evo2_1b_8k_config_path):
    """
    Tests loading weights for the model. Skipped if weights path is not provided.
    """
    weights_path = os.environ.get("EVO2_1B_8K_WEIGHTS_PATH")
    if not weights_path: # Should be caught by skipif, but as a safeguard
        pytest.skip("EVO2_1B_8K_WEIGHTS_PATH environment variable is not set.")

    if not os.path.exists(weights_path):
         pytest.skip(f"Weights file not found at path: {weights_path}")

    try:
        model = load_model(config_path=evo2_1b_8k_config_path, weights_path=weights_path, device="cpu", verbose=False)
        # Basic assertion: model loaded without exceptions
        assert model is not None, "Model loading failed when weights path was provided."
        print(f"Successfully loaded model with weights from {weights_path} for testing.")
    except Exception as e:
        pytest.fail(f"Loading weights failed with an exception: {e}")

# To run these tests:
# 1. Make sure you have pytest installed: pip install pytest
# 2. Navigate to the root directory of your repository.
# 3. Run: pytest
#
# To run the weights test, set the environment variable:
# export EVO2_1B_8K_WEIGHTS_PATH="/path/to/your/weights.pt"
# Then run pytest.
#
# Note on config path:
# The tests assume that 'configs/evo2-1b-8k.yml' is present relative to the root.
# If your test structure is different, adjust the path in CONFIG_PATH_EVO2_1B_8K.
# The evo2_1b_8k_config_path fixture constructs an absolute path.
#
# Note on sys.path modification:
# This is a common way to make modules in the parent directory importable for tests.
# For more complex projects, using package installation (setup.py or pyproject.toml)
# would be more robust.
