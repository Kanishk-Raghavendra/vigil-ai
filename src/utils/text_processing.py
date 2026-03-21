"""Text processing and NLP utilities for VIGIL system."""

import re
from typing import List, Set
import string


class TextProcessor:
    """Utilities for processing and normalizing text."""
    
    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text for processing.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text (lowercase, no extra spaces)
        """
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    @staticmethod
    def extract_nouns(text: str) -> List[str]:
        """
        Simple noun extraction using common patterns.
        
        Does NOT require spaCy dependency - uses heuristics only.
        
        Args:
            text: Input text
            
        Returns:
            List of likely nouns
        """
        # Remove common stop words
        stop_words = {
            'is', 'are', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should',
            'may', 'might', 'must', 'to', 'of', 'in', 'on', 'at', 'by',
            'from', 'with', 'for', 'and', 'or', 'but', 'not', 'the', 'a', 'an'
        }
        
        # Simple heuristic: words after articles or at sentence start
        text_lower = text.lower()
        
        # Find words after "a", "an", "the"
        article_pattern = r'(?:a|an|the)\s+([a-z]+)'
        nouns = re.findall(article_pattern, text_lower)
        
        # Filter stop words and very short words
        nouns = [n for n in nouns if n not in stop_words and len(n) > 2]
        
        return list(set(nouns))  # Deduplicate
    
    @staticmethod
    def extract_attributes(text: str) -> List[str]:
        """
        Extract adjectives/attributes using simple patterns.
        
        Args:
            text: Input text
            
        Returns:
            List of likely attributes
        """
        text_lower = text.lower()
        
        # Common adjective suffixes
        adjective_endings = ('.ing', '.ed', '.ous', '.ful', '.less')
        
        # Extract words that look like adjectives
        word_pattern = r'\b([a-z]+(?:ing|ed|ous|ful|less))\b'
        adjectives = re.findall(word_pattern, text_lower)
        
        # Also get words before nouns (simple heuristic)
        adj_noun_pattern = r'\b([a-z]+)\s+(?:a|an|the)?\s*(?:dog|cat|car|person|building|tree|bird|chair|table)'
        adjectives.extend(re.findall(adj_noun_pattern, text_lower))
        
        return list(set([a for a in adjectives if len(a) > 2]))
    
    @staticmethod
    def extract_relations(text: str) -> List[str]:
        """
        Extract spatial/semantic relations.
        
        Args:
            text: Input text
            
        Returns:
            List of relations found
        """
        text_lower = text.lower()
        
        # Common spatial relations
        spatial_patterns = [
            r'(\w+)\s+(?:on|in|under|above|behind|in front of|next to|near|beside)\s+(?:a|an|the)?\s*(\w+)',
            r'(?:on|in|under|above|behind)\s+(?:a|an|the)?\s*(\w+)',
            r'(?:is|are)\s+(?:on|in|under|above|behind|in front of|next to|near|beside)\s+',
        ]
        
        relations = []
        for pattern in spatial_patterns:
            relations.extend(re.findall(pattern, text_lower))
        
        return relations
