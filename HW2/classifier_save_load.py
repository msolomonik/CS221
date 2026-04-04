#!/usr/bin/python
import argparse
import numpy as np
import pickle
import sys
from util_adjusted import Vocabulary

# number of output prediction classes
K = 6
EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


def _local_softmax(logits: np.ndarray) -> np.ndarray:
    """Local softmax implementation to avoid circular imports."""
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _local_text_to_features(text, vocab) -> np.ndarray:
    """Local text_to_features to avoid circular imports."""
    features = np.zeros(vocab.size())
    for word in text.split():
        features[vocab.get_index(word)] = 1
    return features


def save_weights(weights: np.ndarray, bias: np.ndarray, vocab: Vocabulary, filepath: str):
    """
    Save trained weights, bias, and vocabulary to a file.
    
    @param weights: Trained weights of shape (num_features, K)
    @param bias: Trained bias of shape (1, K)
    @param vocab: Vocabulary object
    @param filepath: Path to save the model
    """
    model_data = {
        'weights': weights,
        'bias': bias,
        'word_to_index': vocab.word_to_index,
        'index_to_word': vocab.index_to_word,
        'next_index': vocab.next_index
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {filepath}")


def load_weights(filepath: str) -> tuple:
    """
    Load trained weights, bias, and vocabulary from a file.
    
    @param filepath: Path to the saved model
    @return: Tuple of (weights, bias, vocab)
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    weights = model_data['weights']
    bias = model_data['bias']
    
    # Reconstruct vocabulary
    vocab = Vocabulary()
    vocab.word_to_index = model_data['word_to_index']
    vocab.index_to_word = model_data['index_to_word']
    vocab.next_index = model_data['next_index']
    
    print(f"Model loaded from {filepath}")
    print(f"Vocabulary size: {vocab.size()}, Weights shape: {weights.shape}")
    
    return weights, bias, vocab


def classify_text(text: str, vocab: Vocabulary, weights: np.ndarray, bias: np.ndarray) -> dict:
    """
    Classify a single text and return predictions with probabilities.
    
    @param text: Input text string
    @param vocab: Vocabulary object
    @param weights: Trained weights
    @param bias: Trained bias
    @return: Dictionary with predicted class, probabilities, and logits
    """
    # Convert text to features
    features = _local_text_to_features(text, vocab)
    features = features.reshape(1, -1)  # Add batch dimension
    
    # Forward pass
    logits = features @ weights + bias
    probabilities = _local_softmax(logits)[0]  # Remove batch dimension
    
    # Get prediction
    predicted_class_idx = int(np.argmax(probabilities))
    predicted_class = EMOTION_LABELS[predicted_class_idx]
    
    # Create probability distribution
    prob_dist = {EMOTION_LABELS[i]: float(probabilities[i]) for i in range(K)}
    
    return {
        'text': text,
        'predicted_class': predicted_class,
        'predicted_class_idx': predicted_class_idx,
        'probabilities': prob_dist,
        'logits': logits[0].tolist()
    }


def interactive_mode(weights: np.ndarray, bias: np.ndarray, vocab: Vocabulary):
    """
    Run interactive classification mode where user can input text.
    
    @param weights: Trained weights
    @param bias: Trained bias
    @param vocab: Vocabulary object
    """
    print("\n" + "="*50)
    print("Interactive Emotion Classifier")
    print("Type 'quit' or 'exit' to stop")
    print("="*50 + "\n")
    
    while True:
        try:
            text = input("Enter text to classify: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                continue
            
            result = classify_text(text, vocab, weights, bias)
            
            print(f"\n  Text: {result['text']}")
            print(f"  Predicted emotion: {result['predicted_class'].upper()}")
            print("  Probabilities:")
            for emotion, prob in sorted(result['probabilities'].items(), 
                                        key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 20)
                print(f"    {emotion:>8}: {prob:.4f} {bar}")
            print()
            
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def batch_classify_file(filepath: str, weights: np.ndarray, bias: np.ndarray, vocab: Vocabulary):
    """
    Classify multiple texts from a file (one per line).
    
    @param filepath: Path to file with texts
    @param weights: Trained weights
    @param bias: Trained bias
    @param vocab: Vocabulary object
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"\nClassifying {len(texts)} texts from {filepath}:\n")
    
    correct = 0
    for i, text in enumerate(texts, 1):
        result = classify_text(text, vocab, weights, bias)
        print(f"{i}. {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"   -> {result['predicted_class'].upper()}")
        print()
    
    print(f"Finished classifying {len(texts)} texts.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Save/load classifier weights and run inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train and save a model:
  python submission_adjusted.py --model linear --lr 1.5
  # Then in Python: save_weights(weights, bias, vocab, 'model.pkl')
  
  # Load and run interactive mode:
  python classifier_save_load.py --load model.pkl --interactive
  
  # Classify texts from a file:
  python classifier_save_load.py --load model.pkl --file texts.txt
  
  # Classify a single text:
  python classifier_save_load.py --load model.pkl --text "I am so happy today!"
        """)
    
    parser.add_argument('--load', type=str, required=True,
                        help='Path to saved model file (.pkl)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive mode for text input')
    parser.add_argument('--file', type=str,
                        help='Path to file with texts to classify (one per line)')
    parser.add_argument('--text', type=str,
                        help='Single text to classify')
    
    args = parser.parse_args()
    
    # Load the model
    weights, bias, vocab = load_weights(args.load)
    
    # Determine mode
    if args.interactive:
        interactive_mode(weights, bias, vocab)
    elif args.file:
        batch_classify_file(args.file, weights, bias, vocab)
    elif args.text:
        result = classify_text(args.text, vocab, weights, bias)
        print(f"\nText: {result['text']}")
        print(f"Predicted emotion: {result['predicted_class'].upper()}")
        print("\nAll probabilities:")
        for emotion, prob in sorted(result['probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {prob:.4f}")
    else:
        print("Error: Specify one of --interactive, --file, or --text")
        sys.exit(1)