"""
Claim Verification Module - VIGIL System

This module handles loading and using CLIP model for verifying claims
by computing similarity between image and text representations.
"""

import torch
import clip
from PIL import Image


class ClaimVerifier:
    """
    Load and use CLIP model for verifying claims.
    CLIP model is loaded once to avoid redundant loading.
    """
    
    def __init__(self):
        """Initialize CLIP model."""
        # Detect device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ClaimVerifier] Using device: {self.device}")
        
        try:
            # Load CLIP model (ViT-B/32 for lightweight inference)
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.model.eval()  # Set to evaluation mode
            print("[ClaimVerifier] CLIP model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")
    
    def verify_claim(self, image_path: str, claim: str) -> float:
        """
        Verify a claim by computing image-text similarity using CLIP.
        
        Args:
            image_path: Path to the input image file
            claim: Text claim to verify
            
        Returns:
            float: Similarity score between 0.0 and 1.0
                   Higher score means higher confidence in the claim
                   
        Raises:
            FileNotFoundError: If image file does not exist
            RuntimeError: If verification fails
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found at: {image_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
        
        try:
            # Encode image and text using CLIP
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                
                # Prepare text - add context for better matching
                text_input = clip.tokenize([claim]).to(self.device)
                text_features = self.model.encode_text(text_input)
            
            # Normalize features for cosine similarity
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity (dot product of normalized vectors = cosine similarity)
            similarity = (image_features @ text_features.T).item()
            
            # Scale to 0-1 range (CLIP similarity is typically in -1 to 1 range)
            # We map it to 0-1 by shifting and scaling
            similarity_normalized = (similarity + 1) / 2
            
            return similarity_normalized
        except Exception as e:
            raise RuntimeError(f"Failed to verify claim: {e}")
