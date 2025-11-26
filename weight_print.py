import torch

def print_all_weights(model):
    """
    Print weights for all layers (quantized or not)
    """
    for name, module in model.named_modules():
        # Check for any layer with weights
        if hasattr(module, 'weight') and module.weight is not None:
            
            print(f"\n{'='*80}")
            print(f"Layer: {name}")
            print(f"Type: {type(module).__name__}")
            print(f"{'='*80}")
            
            # Try to get weight
            try:
                # For quantized layers
                if hasattr(module.weight, '__call__'):
                    qweight = module.weight()
                    int_repr = qweight.int_repr()
                    print(f"\nQuantized Weights (INT8):")
                    print(int_repr)
                else:
                    # For regular layers
                    weight = module.weight
                    print(f"\nWeights (FP32):")
                    print(weight)
            except Exception as e:
                print(f"Could not access weight: {e}")
            
            # Print bias if exists
            try:
                if hasattr(module, 'bias') and module.bias is not None:
                    if hasattr(module.bias, '__call__'):
                        bias = module.bias()
                    else:
                        bias = module.bias
                    print(f"\nBias:")
                    print(bias)
            except Exception as e:
                print(f"Could not access bias: {e}")
        
        # Check for packed parameters (quantized Linear layers)
        if hasattr(module, '_packed_params'):
            print(f"\n{'='*80}")
            print(f"Layer: {name}")
            print(f"Type: {type(module).__name__} (Packed)")
            print(f"{'='*80}")
            
            try:
                # Unpack parameters
                weight, bias = module._weight_bias()
                
                # Get integer representation
                int_repr = weight.int_repr()
                print(f"\nQuantized Weights (INT8):")
                print(int_repr)
                
                if bias is not None:
                    print(f"\nBias:")
                    print(bias)
            except Exception as e:
                print(f"Could not unpack parameters: {e}")

def main():
    # Load model
    print("Loading model...")
    quant_model = torch.jit.load("vgg6_int8.pt", map_location="cpu")
    quant_model.eval()
    
    print("\n" + "="*80)
    print("EXTRACTING WEIGHTS")
    print("="*80)
    
    # Print all weights
    print_all_weights(quant_model)

if __name__ == "__main__":
    main()