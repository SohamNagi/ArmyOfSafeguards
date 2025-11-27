"""
Granite Guardian Wrapper for Army of Safeguards.

This module provides a wrapper around IBM Granite Guardian 3.3 8B model
that is compatible with the benchmark evaluation framework.

Granite Guardian is a specialized model for judging if prompts and responses
meet specified safety criteria (jailbreak, profanity, hallucinations, etc.).
"""

import os
import re
from typing import Dict, Any, Optional, Tuple
import warnings

# Try to import required dependencies
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available. Install with: pip install transformers")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("torch not available. Install with: pip install torch")

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    # Fallback to transformers pipeline if vLLM not available
    try:
        from transformers import pipeline
        PIPELINE_AVAILABLE = True
    except ImportError:
        PIPELINE_AVAILABLE = False

# Model configuration
MODEL_NAME = "ibm-granite/granite-guardian-3.3-8b"
SAFE_TOKEN = "no"  # "no" means safe
RISKY_TOKEN = "yes"  # "yes" means unsafe

# Global model and tokenizer instances (lazy loading)
_model = None
_tokenizer = None
_pipeline = None


def _parse_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse Granite Guardian response to extract score and reasoning trace.
    
    Args:
        response: Raw model output
        
    Returns:
        Tuple of (score, trace) where score is 'yes' or 'no', trace is reasoning (if available)
    """
    # Granite Guardian uses <think> or <think> tags for reasoning
    # and <score> tags for the final judgment
    trace_match = re.findall(r'<(?:redacted_reasoning|think)>(.*?)</(?:redacted_reasoning|think)>', response, re.DOTALL)
    score_match = re.findall(r'<score>(.*?)</score>', response, re.DOTALL)
    
    score, trace = None, None
    
    if trace_match:
        trace = trace_match[-1].strip()
    
    if score_match:
        score = score_match[-1].strip().lower()
    
    return score, trace


def _load_model_vllm(device: str = "cuda"):
    """Load model using vLLM (faster inference)."""
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM not available. Install with: pip install vllm")
    
    print(f"Loading Granite Guardian model using vLLM...")
    
    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model with vLLM
    _model = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        trust_remote_code=True
    )
    
    print("✅ Model loaded successfully")
    return _model, _tokenizer


def _load_model_pipeline(device: str = "cuda"):
    """Load model using transformers pipeline (fallback if vLLM not available)."""
    global _pipeline, _tokenizer
    
    if _pipeline is not None and _tokenizer is not None:
        return _pipeline, _tokenizer
    
    if not PIPELINE_AVAILABLE:
        raise ImportError("transformers pipeline not available. Install with: pip install transformers")
    
    print(f"Loading Granite Guardian model using transformers pipeline...")
    
    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Load model with pipeline
    device_map = "auto" if device == "cuda" and torch.cuda.is_available() else "cpu"
    _pipeline = pipeline(
        "text-generation",
        model=MODEL_NAME,
        tokenizer=_tokenizer,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
    )
    
    print("✅ Model loaded successfully")
    return _pipeline, _tokenizer


def _load_model(device: str = "cuda", prefer_vllm: bool = True):
    """
    Load the Granite Guardian model.
    
    Args:
        device: Device to use ("cuda" or "cpu")
        prefer_vllm: Whether to prefer vLLM over transformers pipeline
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if prefer_vllm and VLLM_AVAILABLE:
        try:
            return _load_model_vllm(device)
        except Exception as e:
            warnings.warn(f"Failed to load with vLLM: {e}. Falling back to transformers pipeline.")
            return _load_model_pipeline(device)
    else:
        return _load_model_pipeline(device)


def _generate_with_vllm(model, tokenizer, chat: str, think: bool = False) -> str:
    """Generate response using vLLM."""
    nlogprobs = 20
    max_tokens = 2048
    
    sampling_params = SamplingParams(
        temperature=0.0,
        logprobs=nlogprobs,
        max_tokens=max_tokens
    )
    
    output = model.generate(chat, sampling_params, use_tqdm=False)
    response = output[0].outputs[0].text.strip()
    return response


def _generate_with_pipeline(pipeline, tokenizer, chat: str, think: bool = False) -> str:
    """Generate response using transformers pipeline."""
    max_tokens = 2048
    
    outputs = pipeline(
        chat,
        max_new_tokens=max_tokens,
        temperature=0.0,
        do_sample=False,
        return_full_text=False
    )
    
    # Extract generated text
    if isinstance(outputs, list) and len(outputs) > 0:
        response = outputs[0].get('generated_text', '')
    else:
        response = str(outputs)
    
    return response.strip()


def evaluate_text(
    text: str,
    threshold: float = 0.5,
    criteria: str = "jailbreak",
    think_mode: bool = False,
    device: str = "cuda",
    documents: Optional[list] = None,
    tools: Optional[list] = None
) -> Dict[str, Any]:
    """
    Evaluate text using Granite Guardian model.
    
    This function is compatible with the benchmark framework and follows
    the same interface as the aggregator evaluate_text functions.
    
    Args:
        text: Input text to evaluate
        threshold: Confidence threshold for flagging (not used directly, but kept for compatibility)
        criteria: Safety criterion to check. Options:
            - "jailbreak": Detect jailbreak attempts
            - "profanity": Detect profane language
            - "function_call": Detect tool-call hallucinations (requires tools parameter)
            - "groundedness": Check if response is grounded in context (requires documents parameter)
        think_mode: Enable thinking mode for detailed reasoning traces
        device: Device to run on ("cuda" or "cpu")
        documents: List of documents for groundedness checking (format: [{'doc_id': str, 'text': str}])
        tools: List of tools for function_call checking (format: [{'name': str, 'description': str, ...}])
        
    Returns:
        Dictionary containing:
            - is_safe: Boolean indicating if content is safe
            - average_confidence: Confidence score (1.0 if unsafe, 0.0 if safe)
            - flags: List of flags (empty if safe, contains one flag if unsafe)
            - individual_results: Dictionary with detailed results
            - score: Raw score from model ("yes" or "no")
            - trace: Reasoning trace (if think_mode enabled)
    """
    # Check dependencies
    if not TRANSFORMERS_AVAILABLE:
        return {
            "is_safe": True,
            "average_confidence": 0.0,
            "flags": [],
            "individual_results": {"error": "transformers not available"},
            "score": None,
            "trace": None
        }
    
    # Validate criteria
    valid_criteria = ["jailbreak", "profanity", "function_call", "groundedness"]
    if criteria not in valid_criteria:
        warnings.warn(f"Unknown criteria '{criteria}'. Using 'jailbreak' instead.")
        criteria = "jailbreak"
    
    # Validate criteria-specific requirements
    if criteria == "function_call" and tools is None:
        warnings.warn("function_call criteria requires tools parameter. Using empty list.")
        tools = []
    
    if criteria == "groundedness" and documents is None:
        warnings.warn("groundedness criteria requires documents parameter. Using empty list.")
        documents = []
    
    try:
        # Load model (lazy loading)
        model, tokenizer = _load_model(device=device, prefer_vllm=True)
        
        # Prepare messages
        messages = [{"role": "user", "content": text}]
        
        # For groundedness, we need assistant response
        if criteria == "groundedness":
            # If text contains both prompt and response, split them
            # Otherwise, assume text is the response
            messages = [{"role": "assistant", "content": text}]
        
        # Prepare guardian config
        guardian_config = {"criteria_id": criteria}
        
        # Prepare chat template
        chat_kwargs = {
            "guardian_config": guardian_config,
            "think": think_mode,
            "tokenize": False,
            "add_generation_prompt": True
        }
        
        # Add optional parameters
        if documents is not None:
            chat_kwargs["documents"] = documents
        if tools is not None:
            chat_kwargs["available_tools"] = tools
        
        # Apply chat template
        chat = tokenizer.apply_chat_template(messages, **chat_kwargs)
        
        # Generate response
        # Check if model is vLLM LLM instance (check by type name to avoid import issues)
        is_vllm = VLLM_AVAILABLE and hasattr(model, 'llm_engine')
        
        if is_vllm:
            response_text = _generate_with_vllm(model, tokenizer, chat, think_mode)
        else:
            response_text = _generate_with_pipeline(model, tokenizer, chat, think_mode)
        
        # Parse response
        score, trace = _parse_response(response_text)
        
        # Convert score to safety assessment
        # "yes" = unsafe, "no" = safe
        is_unsafe = (score == RISKY_TOKEN)
        is_safe = not is_unsafe
        
        # Calculate confidence
        # For Granite Guardian, we use binary confidence:
        # - If unsafe (yes): confidence = 1.0
        # - If safe (no): confidence = 0.0
        # This is a simplification; in practice, you might want to extract
        # logprobs or use the model's confidence scores if available
        confidence = 1.0 if is_unsafe else 0.0
        
        # Create flags list
        flags = []
        if is_unsafe:
            flags.append({
                "safeguard": "granite_guardian",
                "criteria": criteria,
                "score": score,
                "confidence": confidence
            })
        
        # Prepare individual results
        individual_results = {
            "granite_guardian": {
                "criteria": criteria,
                "score": score,
                "is_safe": is_safe,
                "confidence": confidence,
                "trace": trace if think_mode else None
            }
        }
        
        return {
            "is_safe": is_safe,
            "average_confidence": confidence,
            "flags": flags,
            "individual_results": individual_results,
            "score": score,
            "trace": trace
        }
        
    except Exception as e:
        warnings.warn(f"Error evaluating text with Granite Guardian: {e}")
        return {
            "is_safe": True,  # Default to safe on error
            "average_confidence": 0.0,
            "flags": [],
            "individual_results": {"error": str(e)},
            "score": None,
            "trace": None
        }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        test_text = " ".join(sys.argv[1:])
    else:
        test_text = input("Enter text to evaluate: ")
    
    print("\nRunning Granite Guardian...")
    print("=" * 60)
    
    result = evaluate_text(test_text, criteria="jailbreak")
    
    print(f"\nOverall Safety: {'✅ SAFE' if result['is_safe'] else '⚠️  FLAGGED'}")
    print(f"Score: {result.get('score', 'N/A')}")
    print(f"Confidence: {result['average_confidence']:.2%}")
    
    if result['flags']:
        print(f"\nFlags ({len(result['flags'])}):")
        for flag in result['flags']:
            print(f"  - {flag['safeguard']} ({flag['criteria']}): {flag['score']}")
    
    if result.get('trace'):
        print(f"\nReasoning Trace:")
        print(result['trace'][:500] + "..." if len(result['trace']) > 500 else result['trace'])
    
    print("=" * 60)

