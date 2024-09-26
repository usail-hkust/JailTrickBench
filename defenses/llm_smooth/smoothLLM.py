# attempt at implementing the SmoothLLM logic from https://arxiv.org/pdf/2310.03684.pdf
# """Perturbation step. The first ingredient in our approach is to randomly perturb prompts passed as input to
# the LLM. Given an alphabet A, we consider three kinds of perturbations:
# • Insert: Randomly sample q% of the characters in P, and after each of these characters, insert a new
# character sampled uniformly from A.
# • Swap: Randomly sample q% of the characters in P, and then swap the characters at those locations by
# sampling new characters uniformly from A.
# • Patch: Randomly sample d consecutive characters in P, where d equals q% of the characters in P, and
# then replace these characters with new characters sampled uniformly from A"""

import random
import string


def random_insert_updated(text, insert_pct):
    """Randomly insert new chars into text after selected characters."""
    num_inserts = int(len(text) * insert_pct)
    indices = random.sample(range(len(text)), num_inserts)
    for idx in sorted(indices, reverse=True):
        new_char = random.choice(string.printable)
        text = text[:idx + 1] + new_char + text[idx + 1:]
    return text


def random_swap_updated(text, swap_pct):
    """Randomly swap chars within the text with new characters."""
    num_swaps = int(len(text) * swap_pct)
    indices = random.sample(range(len(text)), num_swaps)
    for i in indices:
        new_char = random.choice(string.printable)
        text = text[:i] + new_char + text[i+1:]
    return text


def random_patch(text, patch_pct):
    """Replace a random contiguous patch."""
    patch_len = int(len(text) * patch_pct)
    start_idx = random.randint(0, len(text)-patch_len)
    patch_str = ''.join(random.choice(string.printable) for _ in range(patch_len))
    text = text[:start_idx] + patch_str + text[start_idx+patch_len:]
    return text


def adaptive_perturb_pct(text, base_pct, min_len=10, max_len=100):
    """Adapt perturbation percentage based on text length."""
    text_len = len(text)
    if text_len <= min_len:
        return base_pct / 2
    elif text_len >= max_len:
        return base_pct * 2
    else:
        return base_pct

