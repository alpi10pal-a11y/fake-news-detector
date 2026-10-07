"""
ml/preprocess.py — shared text-cleaning utility.

Imported by both ml/train.py (during training) and backend/routers/predict.py
(at inference time) to guarantee identical preprocessing in both paths.
"""

import re
import nltk

# Download stopwords corpus on first use; no-op if already present.
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

_STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Clean a raw news article string.

    Steps:
      1. Lowercase
      2. Remove all non-alphabetic characters (digits, punctuation, symbols)
      3. Tokenise by splitting on whitespace
      4. Remove English stopwords
      5. Rejoin tokens with a single space

    Returns an empty string if the input is empty or contains only noise.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Keep only alphabetic characters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # 3 & 4. Tokenise and remove stopwords in one pass
    tokens = [word for word in text.split() if word and word not in _STOPWORDS]

    # 5. Rejoin
    return " ".join(tokens)
