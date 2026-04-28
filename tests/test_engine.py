import sys
import os
import pytest

# Add the project root to sys.path so we can import Chess_AI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Chess_AI.ChessEngine import GameState, Move

def test_initial_state():
    gs = GameState()
    assert gs.whiteToMove == True
    assert len(gs.moveLog) == 0
    assert gs.whiteKingLocation == (7, 4)
    assert gs.blackKingLocation == (0, 4)

def test_pawn_moves():
    gs = GameState()
    moves = gs.getValidMoves()
    assert len(moves) == 20
    
    move = gs.uci_to_move("e2e4")
    gs.makeMove(move)
    assert gs.board[4][4] == "wP"
    assert gs.whiteToMove == False

def test_checkmate():
    gs = GameState()
    # Fool's mate
    for m in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        gs.makeMove(gs.uci_to_move(m))
    assert gs.checkmate == True

def test_castling_simple():
    gs = GameState()
    # Remove pieces for kingside castle
    gs.board[7][5] = "--" # f1
    gs.board[7][6] = "--" # g1
    # Ensure rights are there
    gs.whiteCastleKingside = True
    # King at e1 (7,4), Rook at h1 (7,7)
    moves = gs.getValidMoves()
    castle_moves = [m for m in moves if m.castle and m.endCol == 6]
    assert len(castle_moves) == 1

def test_en_passant():
    gs = GameState()
    gs.load_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1")
    moves = gs.getValidMoves()
    ep_moves = [m for m in moves if m.enPassant]
    assert len(ep_moves) == 1
    gs.makeMove(ep_moves[0])
    assert gs.board[2][3] == "wP"
    assert gs.board[3][3] == "--"

def test_undo_move():
    gs = GameState()
    gs.makeMove(gs.uci_to_move("e2e4"))
    gs.undoMove()
    assert len(gs.moveLog) == 0
    assert gs.whiteToMove == True
