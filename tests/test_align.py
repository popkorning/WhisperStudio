"""Forced alignment gives words the aligner has no letters for (digits, Latin) their own time.

CPU only, numpy, no model. Run from the repo root:

    python -m unittest tests.test_align -v
"""
from __future__ import annotations

import unittest

import numpy as np

from whisper_studio import align


def emissions(frames: list[int], classes: int) -> np.ndarray:
    """Log-probabilities where frame t is sure of class frames[t]."""
    em = np.full((len(frames), classes), np.log(0.01 / classes))
    for t, c in enumerate(frames):
        em[t, c] = np.log(0.99)
    return em


class Wildcard(unittest.TestCase):
    def test_unknown_word_gets_its_frames(self):
        blank, a, b, x, y = 0, 1, 2, 3, 4  # x and y: letters the transcript can't name
        frames = [a] * 3 + [blank] + [x] * 3 + [blank] + [y] * 3 + [blank] + [x] * 3 + [blank] + [b] * 3
        em = emissions(frames, 5)
        letters = np.delete(em, [blank], axis=1)
        em = np.concatenate([em, letters.max(axis=1, keepdims=True)], axis=1)
        star = em.shape[1] - 1
        spans = align._viterbi(em, [a, star, star, star, b], blank)
        self.assertIsNotNone(spans)
        self.assertEqual((spans[1][0], spans[3][1]), (4, 14))  # the unknown word covers its sound
        self.assertEqual(spans[-1][0], 16)

    def test_spoken_length_of_numbers(self):
        self.assertLess(align.spoken_len("10"), align.spoken_len("25"))
        self.assertLess(align.spoken_len("25"), align.spoken_len("32500"))
        self.assertGreaterEqual(align.spoken_len("0"), 3)


if __name__ == "__main__":
    unittest.main()
