from .preprocessor import TextPreProcessor
import pick

# preprocessor = TextPreProcessor()
# text = "Hello, World! This is a test. Let's see how it works."
# print(preprocessor.clean(text))

# #python -m core.test_preprocess
# corpus = preprocessor.load_corpus('data/raw/trec-covid/corpus.jsonl', limit=5)
# print(len(corpus))
# print(corpus[0])

scores = [0.5, 4.2, 1.3, 2.8]

sorted(range(4), key=lambda i: scores[i], reverse=True)
# خروجی: [1, 3, 2, 0]
# یعنی: index 1 (4.2) بزرگترین، index 0 (0.5) کوچکترین