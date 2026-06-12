
import json


class TextPreProcessor:
    
    
    
    def text_lower(self, text: str) -> str:
        return text.lower()
    
    def remove_punctuation(self, text: str) -> str:
        return ''.join(char for char in text if char.isalnum() or char.isspace())
    
    def tokenization(self, text: str) -> list:
        return text.split()
    
    def clean(self, text: str) -> list:
        text = self.text_lower(text)
        text = self.remove_punctuation(text)
        text = self.tokenization(text)
        return text

    
    def load_corpus(self, file_path: str , limit : int = None) -> list:
        with open(file_path, 'r', encoding='utf-8') as file:
            result = []
            for i, line in enumerate(file):
                if limit is not None and i >= limit:
                    break
                result.append(json.loads(line))
        
        return result
        