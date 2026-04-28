import random


class OpeningBookNode:
    def __init__(self):
        self.children = {}
        self.weight = 1.0


# Each entry: ([uci_moves_list], weight)
OPENINGS = {
    "Sicilian Najdorf": [
        (["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"], 3.0),
        (["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6", "c1e3"], 2.0),
        (["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6", "f2f3"], 1.5),
        (["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6", "f1e2"], 1.0),
    ],
    "Queen's Gambit": [
        (["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"], 2.0),
        (["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "g1f3", "f8e7"], 2.0),
        (["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "d5c4"], 1.5),
        (["d2d4", "d7d5", "c2c4", "e7e6", "g1f3", "g8f6", "b1c3", "c7c5"], 1.0),
    ],
    "King's Indian": [
        (["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3", "e8g8"], 2.0),
        (["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "f2f3"], 1.5),
        (["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3", "e8g8", "f1e2"], 1.0),
    ],
    "London System": [
        (["d2d4", "d7d5", "g1f3", "g8f6", "c1f4", "e7e6", "e2e3", "f8d6"], 2.0),
        (["d2d4", "g8f6", "g1f3", "d7d5", "c1f4", "c7c5", "e2e3", "b1c3"], 1.5),
        (["d2d4", "d7d5", "g1f3", "g8f6", "c1f4", "c7c5", "c2c3", "b1d2"], 1.0),
    ],
    "Ruy Lopez": [
        (["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"], 2.0),
        (["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7", "f1e1"], 1.5),
        (["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6"], 1.0),
    ],
    "Italian Game": [
        (["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6"], 2.0),
        (["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "d2d3", "g8f6"], 1.5),
        (["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "b2b4"], 1.0),
    ],
}


class OpeningBook:
    def __init__(self, opening_name="Sicilian Najdorf"):
        self.root = OpeningBookNode()
        self.opening_name = opening_name
        lines = OPENINGS.get(opening_name, [])
        for moves, weight in lines:
            self._insert(moves, weight)

    def _insert(self, moves, weight):
        node = self.root
        for m in moves:
            if m not in node.children:
                node.children[m] = OpeningBookNode()
            node = node.children[m]
        node.weight = weight

    def lookup(self, history):
        """
        Walk the trie along the played move history.
        Returns list of (uci_move, weight) for next book moves,
        or None if position is out of book.
        """
        node = self.root
        for m in history:
            if m not in node.children:
                return None    # out of book — hand off to engine
            node = node.children[m]
        if not node.children:
            return None        # end of all lines — hand off to engine
        return [(mv, child.weight) for mv, child in node.children.items()]