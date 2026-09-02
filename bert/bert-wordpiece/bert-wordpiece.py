from typing import List, Dict

class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into WordPiece tokens.
        """
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subwords using greedy longest-match-first.
        """
        # Constraint: If the word exceeds max length, replace the whole word with [UNK]
        if len(word) > self.max_word_len:
            return [self.unk_token]
            
        sub_tokens = []
        start = 0
        
        # Traverse the word to find subword matches
        while start < len(word):
            end = len(word)
            match_found = False
            
            # Shrink the end boundary until we find a match in the vocabulary
            while end > start:
                sub_str = word[start:end]
                
                # If this isn't the first piece of the word, add the continuation prefix
                if start > 0:
                    sub_str = "##" + sub_str
                    
                # Check if the piece exists in our vocabulary
                if sub_str in self.vocab:
                    sub_tokens.append(sub_str)
                    start = end  # Move the start pointer to the end of the matched piece
                    match_found = True
                    break  # Break out of the shrinking `end` loop
                else:
                    end -= 1
            
            # If we shrunk all the way to `start` without finding a match, the word is un-tokenizable
            if not match_found:
                return [self.unk_token]
                
        return sub_tokens