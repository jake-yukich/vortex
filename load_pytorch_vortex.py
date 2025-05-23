import torch
import yaml
import argparse
from typing import Optional, Dict, Any

# Assuming the vortex module is in the Python path or a way to import it is set up
# e.g., if this script is in the root and 'vortex' is a directory in the root.
from vortex.model_pytorch.model import StripedHyena

class SimpleConfig:
    """
    A simple configuration wrapper that takes a dictionary and provides attribute-style access.
    """
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        """Return None for missing keys to avoid errors for less critical params."""
        return self.__dict__.get(name)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Provides a get method similar to dictionaries."""
        return self.__dict__.get(key, default)

def load_model(config_path: str, weights_path: Optional[str] = None, device: str = 'cpu') -> StripedHyena:
    """
    Loads the StripedHyena model from a YAML configuration file and optionally loads weights.

    Args:
        config_path: Path to the YAML configuration file.
        weights_path: Optional path to the model weights checkpoint (.pt or .pth file).
        device: The device to load the model onto ('cpu', 'cuda', 'cuda:0', etc.).
        verbose: If True, print loading messages.

    Returns:
        The loaded StripedHyena model.
    """
    if verbose:
        print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Wrap the dictionary in the SimpleConfig object
    config = SimpleConfig(config_dict)
    
    # Add/override device in config
    config.device = device
    # Ensure use_flash_attn is False as per previous refactoring, if not already in YAML
    config.use_flash_attn = False 
    # Ensure use_flashfft is False for HyenaCascade, if not already in YAML
    config.use_flashfft = False
    # Ensure use_flash_depthwise is False for HyenaCascade, if not already in YAML
    config.use_flash_depthwise = False

    if verbose:
        print(f"Instantiating StripedHyena model on device: {device}")
    # The StripedHyena model's __init__ was modified to accept the device directly
    # and handle internal component placement.
    model = StripedHyena(config) 
    model.to(device) # Ensure top-level model is on the correct device.

    if weights_path:
        if verbose:
            print(f"Loading weights from: {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location=device)
            # If the checkpoint is nested (e.g., under a 'model_state_dict' key)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict: # Common alternative
                state_dict = state_dict['state_dict']
            
            # Use the custom_load_state_dict method if it handles specific checkpoint structures
            # or standard load_state_dict if it's a plain state_dict.
            # Based on previous tasks, custom_load_state_dict was kept.
            model.custom_load_state_dict(state_dict, strict=False) # Use strict=False if some keys might not match
            if verbose:
                print("Weights loaded successfully.")
        except Exception as e:
            if verbose:
                print(f"Error loading weights: {e}. Model will be randomly initialized.")
    else:
        if verbose:
            print("No weights path provided. Model is randomly initialized.")

    model.eval() # Set to evaluation mode
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load a PyTorch-native StripedHyena model.")
    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("--weights_path", type=str, default=None, help="Optional path to the model weights checkpoint.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load the model on (e.g., 'cpu', 'cuda', 'cuda:0').")

    args = parser.parse_args()

    try:
        # When running as a script, enable verbose output
        model = load_model(config_path=args.config_path, weights_path=args.weights_path, device=args.device, verbose=True)
        # print(f"Model loaded successfully on device {args.device}.") # Already printed by load_model if verbose

        # Simple test
        # print("Performing a simple forward pass test...") # Already printed by load_model if verbose
        # Example: Batch size 1, sequence length 100, random vocab indices (0-511)
        # Adjust vocab size (512) if different in your config.
        # The vocab size is typically config.vocab_size
        vocab_size = model.config.get('vocab_size', 512) # Default to 512 if not in config
        input_tensor = torch.randint(0, vocab_size, (1, 100), device=args.device) 
        
        with torch.no_grad(): # Disable gradient calculations for inference
            output_tensor, _ = model.forward(input_tensor) # Assuming forward returns (output, inference_params_dict)
        
        if model.config.get('verbose', True): # Check if verbose is enabled in config or default to True
            print(f"Input tensor shape: {input_tensor.shape}")
            print(f"Output tensor shape: {output_tensor.shape}")
            print("Test forward pass completed.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
