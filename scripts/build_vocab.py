#!/usr/bin/env python3
"""
build_vocab.py

Purpose:
- Simple verification script for the final runtime vocabulary files.
- Loads the beginner, intermediate, and advanced text files.
- Prints their sizes so you can confirm they were created correctly.

This script does NOT build the lists.
It only checks that the final TXT outputs can be loaded and that their lengths
look correct.
"""

# Load a one-word-per-line vocabulary file into a lowercase set.
# A set is useful because it automatically removes duplicates and allows fast lookup.
def load_word_list(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def main():
    """
    Open the three runtime vocabulary files and print their sizes.
    Expected totals in this project:
    - beginner_1000.txt      -> about 1000 words
    - intermediate_3000.txt  -> about 3000 words
    - advanced_6000.txt      -> about 6000 words
    """
    beginner = load_word_list("data/vocab/beginner_1000.txt")
    intermediate = load_word_list("data/vocab/intermediate_3000.txt")
    advanced = load_word_list("data/vocab/advanced_6000.txt")

    print("Beginner Size: ", len(beginner))
    print("Intermediate Size: ", len(intermediate))
    print("Advanced Size: ", len(advanced))


if __name__ == "__main__":
    main()
