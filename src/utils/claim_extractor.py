"""Improved claim extraction for VIGIL system."""

import re
from typing import List, Dict, Any
from src.utils.text_processing import TextProcessor
from src.utils.logger import Logger


logger = Logger(__name__)


class Claim:
    """Data structure representing a claim."""
    
    def __init__(self, text: str, claim_type: str, confidence: float = 1.0):
        """
        Initialize a claim.
        
        Args:
            text: Claim text
            claim_type: Type of claim ('object', 'attribute', 'relation')
            confidence: Type confidence (how confident the extraction is)
        """
        self.text = text
        self.claim_type = claim_type
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert claim to dictionary."""
        return {
            "text": self.text,
            "type": self.claim_type,
            "confidence": self.confidence
        }


class ClaimExtractor:
    """
    Improved claim extraction using pattern matching and NLP heuristics.
    
    Extracts:
    - Object claims: "There is a dog"
    - Attribute claims: "There is a red car"
    - Relation claims: "A dog is on a table"
    """
    
    # Built-in keywords for common objects
    COMMON_OBJECTS = {
        'dog', 'cat', 'car', 'person', 'people', 'building', 'tree', 'bird',
        'chair', 'table', 'phone', 'computer', 'bottle', 'cup', 'flower',
        'house', 'street', 'door', 'window', 'wall', 'floor', 'sky', 'grass',
        'water', 'animal', 'elephant', 'horse', 'bicycle', 'motorcycle',
        'bus', 'train', 'airplane', 'book', 'desk', 'lamp', 'bed', 'sofa'
    }
    
    @staticmethod
    def extract_claims(caption: str) -> List[Claim]:
        """
        Extract claims from caption.
        
        Args:
            caption: Input caption text
            
        Returns:
            List of Claim objects
        """
        claims = []
        caption_normalized = TextProcessor.normalize(caption)
        
        # Extract object claims
        object_claims = ClaimExtractor._extract_object_claims(caption_normalized)
        claims.extend(object_claims)
        
        # Extract attribute claims
        attribute_claims = ClaimExtractor._extract_attribute_claims(caption_normalized)
        claims.extend(attribute_claims)
        
        # Extract relation claims
        relation_claims = ClaimExtractor._extract_relation_claims(caption_normalized)
        claims.extend(relation_claims)
        
        # Deduplicate
        unique_claims = []
        seen = set()
        for claim in claims:
            claim_key = claim.text.lower()
            if claim_key not in seen:
                unique_claims.append(claim)
                seen.add(claim_key)
        
        logger.info(f"Extracted {len(unique_claims)} claims from caption")
        return unique_claims
    
    @staticmethod
    def _extract_object_claims(text: str) -> List[Claim]:
        """Extract object-based claims."""
        claims = []
        text_lower = text.lower()
        
        # Pattern 1: "there is/are a/an {object}"
        pattern1 = r"there (?:is|are) (?:a|an)?\s*(\w+)"
        for match in re.finditer(pattern1, text_lower):
            obj = match.group(1)
            if len(obj) > 2 and obj not in {'is', 'are', 'on', 'in'}:
                claims.append(Claim(f"There is a {obj}", "object", 0.9))
        
        # Pattern 2: "{article} {noun}" at sentence boundaries
        pattern2 = r"(?:^|\.|\s)(?:a|an|the)\s+(\w+)(?:\s+(?:is|are|with|sitting|standing)|\s*\.|\s*$)"
        for match in re.finditer(pattern2, text_lower):
            obj = match.group(1)
            if len(obj) > 2 and obj not in {'is', 'are'} and obj in ClaimExtractor.COMMON_OBJECTS:
                claims.append(Claim(f"There is a {obj}", "object", 0.85))
        
        # Pattern 3: Nouns after action verbs
        pattern3 = r"(?:holding|carrying|wearing|riding|sitting on|standing on|using)\s+(?:a|an)?\s*(\w+)"
        for match in re.finditer(pattern3, text_lower):
            obj = match.group(1)
            if len(obj) > 2:
                claims.append(Claim(f"There is a {obj}", "object", 0.8))
        
        return claims
    
    @staticmethod
    def _extract_attribute_claims(text: str) -> List[Claim]:
        """Extract attribute-based claims."""
        claims = []
        text_lower = text.lower()
        
        # Pattern: "{adjective} {noun}"
        pattern = r"(\w+)\s+(?:dog|cat|car|person|building|tree|bird|chair|table|house|person|animal)\b"
        
        adjective_indicators = {
            'big', 'small', 'large', 'tiny', 'tall', 'short', 'wide', 'narrow',
            'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'purple',
            'pink', 'orange', 'gray', 'silver', 'golden', 'wooden', 'metal',
            'old', 'new', 'ancient', 'modern', 'dirty', 'clean', 'wet', 'dry'
        }
        
        for match in re.finditer(pattern, text_lower):
            adj = match.group(1)
            if adj in adjective_indicators:
                noun = match.group(0).split()[-1]
                claims.append(Claim(f"There is a {adj} {noun}", "attribute", 0.75))
        
        return claims
    
    @staticmethod
    def _extract_relation_claims(text: str) -> List[Claim]:
        """Extract relation-based claims."""
        claims = []
        text_lower = text.lower()
        
        # Spatial relations
        relations = ['on', 'in', 'under', 'above', 'behind', 'in front of', 'next to', 'beside', 'near']
        
        for relation in relations:
            pattern = rf"(\w+(?:\s+\w+)*?)\s+{relation}\s+(?:a|an|the)?\s*(\w+)"
            for match in re.finditer(pattern, text_lower):
                subj = match.group(1).strip()
                obj = match.group(2)
                
                if len(subj) > 2 and len(obj) > 2:
                    claim_text = f"There is a {subj} {relation} a {obj}"
                    claims.append(Claim(claim_text, "relation", 0.7))
        
        return claims
    
    @staticmethod
    def filter_claims(claims: List[Claim], min_confidence: float = 0.0) -> List[Claim]:
        """
        Filter claims by confidence and validity.
        
        Args:
            claims: List of claims to filter
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of claims
        """
        filtered = []
        for claim in claims:
            # Validity checks
            if not claim.text or len(claim.text.strip()) == 0:
                continue
            if len(claim.text) > 100:
                continue
            if claim.confidence < min_confidence:
                continue
            
            filtered.append(claim)
        
        return filtered
