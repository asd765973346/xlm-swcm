# -*- coding: utf-8 -*-
"""
Simple inference example for the Seq2SeqModel with checkpoint loading.

This script demonstrates how to:
1. Load a pre-trained checkpoint
2. Initialize the model
3. Perform basic inference

Author: Your Name
"""

import torch
from transformers import XLMRobertaTokenizer, XLMRobertaConfig
from model import Seq2SeqModel


def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix added by DataParallel.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        dict: Cleaned state dictionary without 'module.' prefix
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' (7 characters)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_checkpoint(model, checkpoint_path):
    """
    Load model checkpoint with proper handling of DataParallel prefixes.
    
    Args:
        model: PyTorch model instance
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        bool: True if loading successful, False otherwise
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Checkpoint loaded from {checkpoint_path}")
        
        # Extract state dictionary
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # Remove module prefix if present
        clean_state_dict = remove_module_prefix(state_dict)
        
        # Load parameters
        missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
        
        print("✓ Model parameters loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False


def main():
    """
    Main function demonstrating model loading and basic usage.
    """
    # Configuration
    model_path = "./pretrained_model/xlm-swcm.bin"
    xlm_model_path = "./base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model parameters
    max_src_len = 256
    max_tgt_len = 256
    batch_size = 1
    teacher_forcing = 0.0  # No teacher forcing during inference
    
    print("=== Initializing Model ===")
    
    # Load configuration
    try:
        model_config = XLMRobertaConfig.from_pretrained(xlm_model_path)
        decoder_config = XLMRobertaConfig.from_dict(model_config.to_dict())
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return
    
    # Initialize model
    try:
        model = Seq2SeqModel(
            model_name_or_path=xlm_model_path,
            decoder_config=decoder_config,
            device=device,
            tgtlen=max_tgt_len,
            batchsize=batch_size,
            teacher_forcing=teacher_forcing
        )
        model.to(device)
        print("✓ Model initialized")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return
    
    # Load checkpoint
    if not load_checkpoint(model, model_path):
        return
    
    # Set model to evaluation mode
    model.eval()
    print("✓ Model set to evaluation mode")
    
    # Load tokenizer
    try:
        tokenizer = XLMRobertaTokenizer.from_pretrained(xlm_model_path)
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return
    
    print("\n=== Model Ready for Inference ===")
    print(f"Device: {device}")
    print(f"Max source length: {max_src_len}")
    print(f"Max target length: {max_tgt_len}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Example usage (optional)
    print("\n=== Example Usage ===")
    sample_text = "འདི་ནི་དཔེ་མཚོན་ཡིག་དེབ་ཞིག་དང་།དཔེ་དབྱིབས་ཀྱི་གཞི་རྩའི་བྱེད་ནུས་གསལ་སྟོན་བྱེད་པར་བཀོལ་།"
    
    # Tokenize input
    inputs = tokenizer(
        sample_text,
        max_length=max_src_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

main()