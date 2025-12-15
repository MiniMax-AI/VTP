"""CLIP tokenizer implementation.

Copied from OpenAI CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import gzip
import html
import os
import string
from functools import lru_cache
from typing import List, Optional, Union

try:
    import ftfy
except ImportError:
    # Fallback: if ftfy is not available, use a simple replacement
    def ftfy_fix_text(text):
        return text
    ftfy = type('ftfy', (), {'fix_text': staticmethod(ftfy_fix_text)})()

try:
    import regex as re
    HAS_REGEX = True
except ImportError:
    import re
    HAS_REGEX = False

import torch

# https://stackoverflow.com/q/62691279
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_CONTEXT_LENGTH = 77  # default context length for OpenAI CLIP


@lru_cache()
def _find_bpe_file():
    """Find BPE file in common locations."""
    possible_paths = [
        # Try in current directory
        os.path.join(os.path.dirname(__file__), "bpe_simple_vocab_16e6.txt.gz"),
        # Try in parent directory
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "bpe_simple_vocab_16e6.txt.gz"),
        # Try in project root
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "bpe_simple_vocab_16e6.txt.gz"),
        # Try in tools directory
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tools", "bpe_simple_vocab_16e6.txt.gz"),
    ]

    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path

    # If not found, return None and let SimpleTokenizer handle the error
    return None


def default_bpe():
    """Get default BPE file path."""
    bpe_path = _find_bpe_file()
    if bpe_path is None:
        # Try to use open_clip's default if available
        try:
            from open_clip.tokenizer import default_bpe as _default_bpe
            return _default_bpe()
        except ImportError:
            raise FileNotFoundError(
                "BPE vocabulary file not found. Please ensure bpe_simple_vocab_16e6.txt.gz "
                "is available in one of the expected locations, or install open_clip."
            )
    return bpe_path


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    """Basic text cleaning."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """Clean whitespace."""
    text = " ".join(text.split())
    text = text.strip()
    return text


def _clean_lower(x):
    """Basic, remove whitespace, lower case."""
    return whitespace_clean(basic_clean(x)).lower()


def _clean_whitespace(x):
    """Basic, remove whitespace."""
    return whitespace_clean(basic_clean(x))


def get_clean_fn(clean_type: str):
    """Get cleaning function by type."""
    if clean_type == 'lower':
        return _clean_lower
    elif clean_type == 'whitespace':
        return _clean_whitespace
    else:
        return _clean_lower  # Default to lower


class SimpleTokenizer(object):
    """Simple tokenizer for CLIP models."""

    def __init__(
            self,
            bpe_path: Optional[str] = None,
            additional_special_tokens: Optional[List[str]] = None,
            context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH,
            clean: str = 'lower',
    ):
        if bpe_path is None:
            bpe_path = default_bpe()

        if not os.path.exists(bpe_path):
            raise FileNotFoundError(
                f"BPE vocabulary file not found at {bpe_path}. "
                "Please ensure the file exists or install open_clip."
            )

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Load BPE merges
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]

        # Build vocabulary
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))

        special_tokens = ['<start_of_text>', '<end_of_text>']
        if additional_special_tokens:
            special_tokens += additional_special_tokens
        vocab.extend(special_tokens)

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t: t for t in special_tokens}

        special = "|".join(special_tokens)
        # Use regex with Unicode support if available, otherwise fallback to standard re
        if HAS_REGEX:
            self.pat = re.compile(
                special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
                re.IGNORECASE,
            )
        else:
            # Fallback pattern if regex doesn't support \p{L} etc.
            self.pat = re.compile(
                special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]+|[^\s\w]+""",
                re.IGNORECASE,
            )

        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]
        self.sot_token_id = self.all_special_ids[0]
        self.eot_token_id = self.all_special_ids[1]
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)

    def bpe(self, token):
        """Apply BPE encoding to a token."""
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        """Encode text to token IDs."""
        bpe_tokens = []
        text = self.clean_fn(text)
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        """Decode token IDs to text."""
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.LongTensor:
        """Returns the tokenized representation of given input string(s).

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length'

        all_tokens = [[self.sot_token_id] + self.encode(text) + [self.eot_token_id] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]  # Truncate
                tokens[-1] = self.eot_token_id
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result


def get_tokenizer(
    model_name: str = 'ViT-B-32',
    context_length: Optional[int] = None,
    cache_dir: Optional[str] = None,
    **kwargs
):
    """
    Get tokenizer instance.

    Args:
        model_name: Model name (for compatibility, not used in simple implementation)
        context_length: Context length for tokenizer
        cache_dir: Cache directory (for compatibility, not used)
        **kwargs: Additional arguments passed to SimpleTokenizer

    Returns:
        SimpleTokenizer instance
    """
    context_length = context_length or DEFAULT_CONTEXT_LENGTH

    # Try to use open_clip's tokenizer if available
    try:
        from open_clip import get_tokenizer as _get_tokenizer
        return _get_tokenizer(model_name, context_length=context_length, cache_dir=cache_dir, **kwargs)
    except ImportError:
        pass

    # Use our simple tokenizer
    return SimpleTokenizer(context_length=context_length, **kwargs)
