class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        # Assign special tokens
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3
        }
        
        # Extract lowercased, whitespace-separated words
        unique_words = set()
        for text in texts:
            unique_words.update(text.lower().split())
            
        # Sort words alphabetically
        sorted_words = sorted(list(unique_words))
        
        # Add to dictionary starting from ID 4
        next_id = 4
        for word in sorted_words:
            self.word_to_id[word] = next_id
            next_id += 1
            
        # Create reverse mapping and set vocab size
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        unk_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(word, unk_id) for word in text.lower().split()]

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        return " ".join([self.id_to_word.get(token_id, self.unk_token) for token_id in ids])