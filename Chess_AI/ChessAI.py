import logging
import random
import chess
from ctypes import cdll, c_char_p, c_int
import os
import torch
import numpy as np

from Chess_AI import predict as pt_100
from Chess_AI import predict_ghost as pt_ghost

pytorch_100_engine = pt_100.predictor
ghost_engine       = pt_ghost.ghost_predictor
nnue = None
base_dir = os.path.dirname(__file__)

# Cross-platform NNUE library loading
if os.name == 'nt':
    lib_path = os.path.join(base_dir, "src", "nnueprobe.dll")
else:
    lib_path = os.path.join(base_dir, "src", "libnnueprobe.so")

try:
    nnue = cdll.LoadLibrary(lib_path)
    nnue.nnue_init.argtypes = [c_char_p]
    nnue.nnue_init.restype = None
    nnue.nnue_evaluate_fen.argtypes = [c_char_p]
    nnue.nnue_evaluate_fen.restype = c_int

    nnue_file_path = os.path.join(base_dir, "src", "nn-04cf2b4ed1da.nnue").encode('utf-8')
    nnue.nnue_init(nnue_file_path)
    logging.info("NNUE loaded and initialized!")
except Exception as e:
    nnue = None
    logging.info(f"[WARN] NNUE library not available ({e}). AI will use material-only evaluation.")

pieceScore = {"K": 0, "Q": 9, "R": 5, "B": 3, "N": 3, "P": 1}

CHECKMATE = 10000
STALEMATE = 0

# ── Killer move heuristic storage ─────────────────────────────────────
# killer_moves[depth] = list of up to 2 killer moves (move objects)
killer_moves = {}

# ── History heuristic table ─────────────────────────────────────────
# history_table[(piece, endRow, endCol)] = score
history_table = {}


# --- Zobrist Hashing ---
def random_64bit():
    return random.getrandbits(64)


zobrist_table = [[[random_64bit() for _ in range(12)] for _ in range(8)] for _ in range(8)]
side_to_move_key = random_64bit()

piece_to_index = {
    "wP": 0, "wN": 1, "wB": 2, "wR": 3, "wQ": 4, "wK": 5,
    "bP": 6, "bN": 7, "bB": 8, "bR": 9, "bQ": 10, "bK": 11
}


def get_zobrist_hash(board, whiteToMove):
    h = 0
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece != "--":
                idx = piece_to_index[piece]
                h ^= zobrist_table[r][c][idx]
    if not whiteToMove:
        h ^= side_to_move_key
    return h

def findModelMovePytorch(gs, validMoves):
    if not validMoves:
        logging.debug("none")
        return None

    # Use get_prediction() — pure PyTorch, handles large negative logits correctly
    probabilities = pytorch_100_engine.get_prediction(gs)

    move_lookup = {move.get_uci_notation(): move for move in validMoves}
    sorted_indices = np.argsort(probabilities)[::-1]

    for idx in sorted_indices:
        move_uci = pytorch_100_engine.int_to_move.get(idx)
        if move_uci in move_lookup:
            rank = int(np.where(sorted_indices == idx)[0][0]) + 1
            top_prob = probabilities[sorted_indices[0]] * 100
            return move_lookup[move_uci]

    return random.choice(validMoves)


# ── Blunder threshold (pawn units) ────────────────────────────────────────
# A candidate move is considered a blunder when its 1-ply NNUE evaluation
# falls more than this many pawns below the GLOBALLY best legal move.
# 1.5 ≈ minor piece value — prevents hanging bishops/knights while still
# allowing the model to play bold, non-trivially-optimal moves.
_GHOST_BLUNDER_THRESHOLD = 1.5


def findModelMoveGhost(gs, validMoves, top_k=10):
    """
    Ghost AI — Medium difficulty  (≈ 1200 ELO)

    Strategy  (hybrid GhostChessNet + global-NNUE blunder checker):

      1. Score ALL legal moves with 1-ply NNUE to find the true global-best
         move available in the position.  This is the only correct baseline
         for "is this a blunder?" — comparing against the model's own top-10
         pool (previous bug) let bad moves slip through when the whole pool
         was bad.

      2. Ask GhostChessNet for a probability distribution and collect the
         top-K candidates (model preference order).

      3. Walk candidates in model-probability order:
           • Accept the first whose NNUE score is within
             _GHOST_BLUNDER_THRESHOLD (1.5 pawns) of the GLOBAL best.
           • Skip moves that fall further than the threshold and log them.

      4. If every model candidate is a blunder (position is losing and the
         model can't find anything good) fall back to the NNUE-best move
         so at least the damage is minimised.

    KEY FIX vs previous version
    ───────────────────────────
    Old code used  max(score for score in TOP-10 pool)  as the reference.
    When the whole pool was mediocre, a bad move could beat a threshold
    measured against an already-bad reference  (e.g. gap 1.487 < 1.5 → accepted).
    New code uses  max(score for ALL legal moves)  so the threshold is always
    anchored to what the engine could actually play.
    """
    if not validMoves:
        return None

    # scoreBoard() is always from White's perspective (positive = White better).
    # turn_mult flips the sign so "higher = better for the side to move".
    turn_mult = 1 if gs.whiteToMove else -1

    # ── Step 1: NNUE-score every legal move (single pass, no redundant calls) ──
    # Result: {uci_string: (move_obj, nnue_score_from_mover_pov)}
    all_scored: dict[str, tuple] = {}
    for move in validMoves:
        gs.makeMove(move)
        score = turn_mult * scoreBoard(gs)
        gs.undoMove()
        all_scored[move.get_uci_notation()] = (move, score)

    # True reference: best score achievable by any legal move
    global_best_score = max(s for _, s in all_scored.values())
    global_best_move  = max(all_scored.values(), key=lambda x: x[1])[0]

    # ── Step 2: ghost model predictions → top-K candidates ──────────────
    probabilities  = ghost_engine.get_prediction(gs)
    move_lookup    = {uci: mv for uci, (mv, _) in all_scored.items()}
    sorted_indices = np.argsort(probabilities)[::-1]

    candidates = []   # [(prob, move_obj, nnue_score), ...]  model-prob order
    for idx in sorted_indices:
        uci = ghost_engine.int_to_move.get(idx)
        if uci in all_scored:
            mv, sc = all_scored[uci]
            candidates.append((probabilities[idx], mv, sc))
        if len(candidates) >= top_k:
            break

    if not candidates:
        # Model produced no recognised legal move at all
        return global_best_move

    # ── Step 3: walk candidates in model-probability order ───────────────
    # Accept the first move that is NOT a blunder vs. the global reference.
    chosen_move  = None
    chosen_score = None
    chosen_prob  = None
    chosen_rank  = None
    blunder_skips = 0

    for rank, (prob, move, score) in enumerate(candidates, start=1):
        gap = global_best_score - score
        if gap <= _GHOST_BLUNDER_THRESHOLD:
            chosen_move  = move
            chosen_score = score
            chosen_prob  = prob
            chosen_rank  = rank
            break
        else:
            blunder_skips += 1


    # ── Step 4: fallback — every model candidate was a blunder ──────────
    if chosen_move is None:

        return global_best_move


    return chosen_move


# def findBestMove(gs, validMoves, position_history=None):
#     """
#     Iterative deepening NegaMax with Alpha-Beta.
#     position_history: dict {zobrist_hash: count} from app.py so the engine
#     can penalise repetitions and avoid drawing from a winning position.
#     """
#     global nextMove, count, DEPTH, transposition_hits, transposition_table, past_moves
#     global killer_moves, history_table
#
#     DEPTH = 4
#     count = 0
#     nextMove = None
#     transposition_table = {}
#     transposition_hits = 0
#     past_moves = {}
#     killer_moves = {}
#     history_table = {}
#
#     score, pieceCount = scoreMaterial(gs.board)
#     if score < 20 or pieceCount < 10:
#         DEPTH += 1
#     if score < 12 or pieceCount < 5:
#         DEPTH += 1
#
#     pos_hist = position_history or {}
#
#     for d in range(1, DEPTH + 1):
#         result = findMoveNegaMaxAlphaBeta(gs, validMoves, d, -CHECKMATE, CHECKMATE,
#                                           1 if gs.whiteToMove else -1,
#                                           root_depth=d, position_history=pos_hist)
#         if abs(result) >= CHECKMATE - 100:
#             logging.info(f"  ✓ Mate found at depth {d}, stopping early.")
#             break
#
#     return nextMove


def findBestMoveAtDepth(gs, validMoves, depth, position_history=None):
    """
    NegaMax with Alpha-Beta at a fixed caller-specified depth.
    Used for difficulty control: depth=2 (~1200 Elo), depth=4 (~1600 Elo).

    Endgame depth boost: when few pieces remain and one side has a decisive
    material advantage, depth is raised automatically so the engine can
    find mate and avoid shuffling into draws from winning positions.
    """
    global nextMove, count, DEPTH, transposition_hits, transposition_table, past_moves
    global killer_moves, history_table

    # ── Adaptive endgame depth ──────────────────────────────────────────
    # scoreMaterial returns (total_material_value, total_piece_count_incl_kings)
    _, piece_count = scoreMaterial(gs.board)

    # Count each side's material to detect lopsided endgames
    white_mat = sum(
        pieceScore.get(gs.board[r][c][1], 0)
        for r in range(8) for c in range(8)
        if gs.board[r][c] != "--" and gs.board[r][c][0] == 'w' and gs.board[r][c][1] != 'K'
    )
    black_mat = sum(
        pieceScore.get(gs.board[r][c][1], 0)
        for r in range(8) for c in range(8)
        if gs.board[r][c] != "--" and gs.board[r][c][0] == 'b' and gs.board[r][c][1] != 'K'
    )
    mat_diff = abs(white_mat - black_mat)

    effective_depth = depth
    if piece_count <= 5:
        # Very simple endgame (e.g. KQ vs K, KR vs K) — always search deep.
        # Alpha-beta prunes so heavily with only a king left that depth 8 is
        # fast (<100ms) and guaranteed to find mate in most KQ/KR endings.
        effective_depth = max(depth, 8)
    elif piece_count <= 8 and mat_diff >= 5:
        # Endgame with decisive material edge (rook or more up) — boost depth
        # enough to see mating patterns and avoid repetition traps.
        effective_depth = max(depth, 6)
    elif piece_count <= 12 and mat_diff >= 5:
        # Late middlegame / early endgame with large advantage.
        effective_depth = max(depth, depth + 2)



    DEPTH = effective_depth
    count = 0
    nextMove = None
    transposition_table = {}
    transposition_hits = 0
    past_moves = {}
    killer_moves = {}
    history_table = {}

    pos_hist = position_history or {}

    for d in range(1, effective_depth + 1):
        result = findMoveNegaMaxAlphaBeta(gs, validMoves, d, -CHECKMATE, CHECKMATE,
                                          1 if gs.whiteToMove else -1,
                                          root_depth=d, position_history=pos_hist)


    return nextMove



def findRandomMove(validMoves):
    return random.choice(validMoves)


def quiescenceSearch(gs, alpha, beta, turnMultiplier):
    global count
    count += 1

    stand_pat = scoreBoard(gs)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    validMoves = gs.getValidMoves()
    capture_moves = [move for move in validMoves if move.pieceCaptured != "--"]

    # MVV-LVA ordering for captures
    piece_values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}

    def capturePriority(move):
        captured_piece = move.pieceCaptured[1].upper()
        attacker_piece = move.pieceMoved[1].upper()
        return piece_values.get(captured_piece, 0) * 10 - piece_values.get(attacker_piece, 0)

    capture_moves.sort(key=capturePriority, reverse=True)

    for move in capture_moves:
        gs.makeMove(move)
        score = -quiescenceSearch(gs, -beta, -alpha, -turnMultiplier)
        gs.undoMove()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


# ── Fast move priority───────
def movePriority(move, gs, depth):

    piece_values = {"K": 0, "Q": 9, "R": 5, "B": 3, "N": 3, "P": 1}
    score = 0

    # 1. Captures — MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
    if move.pieceCaptured != "--":
        victim_value = piece_values.get(move.pieceCaptured[1], 0)
        attacker_value = piece_values.get(move.pieceMoved[1], 0)
        score += 10 * victim_value - attacker_value + 100  # bias captures above quiet moves

    # 2. Pawn promotion
    if move.pawnPromotion:
        score += 90

    # 3. Castling
    if move.castle:
        score += 20

    # 4. Killer moves (quiet moves that caused beta-cutoff at same depth)
    killers = killer_moves.get(depth, [])
    if move in killers:
        score += 80

    # 5. History heuristic (quiet moves that historically improved alpha)
    h_key = (move.pieceMoved, move.endRow, move.endCol)
    score += history_table.get(h_key, 0)

    # 6. Center control
    center_squares = {(3, 3), (3, 4), (4, 3), (4, 4)}
    if (move.endRow, move.endCol) in center_squares and move.pieceMoved[1] in {"P", "N", "B"}:
        score += 3

    # 7. Minor piece development
    if move.pieceMoved[1] in {"N", "B"} and move.startRow in (7, 0):
        score += 3

    return score


def findMoveNegaMaxAlphaBeta(gs, validMoves, depth, alpha, beta, turnMultiplier,
                              root_depth=None, position_history=None):
    global count, nextMove, transposition_hits, transposition_table, past_moves
    global killer_moves, history_table, DEPTH

    if root_depth is None:
        root_depth = DEPTH
    if position_history is None:
        position_history = {}

    count += 1
    zobrist_hash = get_zobrist_hash(gs.board, gs.whiteToMove)

    # ── Repetition penalty: if this position has been seen before, discourage it ──
    # A position seen twice before is almost threefold — heavily penalise it.
    # A position seen once is mildly penalised to discourage shuffling.
    repeat_count = position_history.get(zobrist_hash, 0)
    if repeat_count >= 2:
        # Returning 0 (draw score) here stops the engine from walking into a repetition.
        # The side to move loses the chance to improve — correctly scored as a draw.
        return STALEMATE
    elif repeat_count == 1:
        # One prior visit: apply a small penalty so the engine avoids it unless forced.
        repetition_penalty = 0.5 * turnMultiplier  # small nudge away from repetition
    else:
        repetition_penalty = 0.0

    # ── Transposition table lookup — also grab PV move for ordering ──
    tt_move = None
    tt_entry = transposition_table.get(zobrist_hash)
    if tt_entry is not None:
        stored_score, stored_depth, tt_flag, tt_move = tt_entry
        if stored_depth >= depth and depth != root_depth:
            if tt_flag == 'EXACT':
                transposition_hits += 1
                return stored_score
            elif tt_flag == 'LOWER' and stored_score >= beta:
                transposition_hits += 1
                return stored_score
            elif tt_flag == 'UPPER' and stored_score <= alpha:
                transposition_hits += 1
                return stored_score

    # ── Leaf node: quiescence search ──
    if depth == 0:
        raw = quiescenceSearch(gs, alpha, beta, turnMultiplier)
        return raw - repetition_penalty

    maxScore = -CHECKMATE
    original_alpha = alpha
    best_move_this_node = None

    # ── Move ordering: PV/TT move first, then static scorer ──
    def sort_key(move):
        if tt_move is not None and move == tt_move:
            return 999999
        return movePriority(move, gs, depth)

    validMoves.sort(key=sort_key, reverse=True)

    # ── Track this position as visited during this search path ──
    position_history[zobrist_hash] = position_history.get(zobrist_hash, 0) + 1

    for move_idx, move in enumerate(validMoves):
        gs.makeMove(move)
        nextMoves = gs.getValidMoves()

        if len(nextMoves) == 0:
            if gs.inCheck:
                # Checkmate: closer mates score higher (subtract depth so engine prefers faster mate)
                score = CHECKMATE - (root_depth - depth)
            else:
                score = STALEMATE
        else:
            # ── Late Move Reduction (LMR) ──
            if (depth >= 3 and move_idx >= 4
                    and move.pieceCaptured == "--"
                    and not move.pawnPromotion
                    and not gs.inCheck):
                reduction = 1 if move_idx < 8 else 2
                score = -findMoveNegaMaxAlphaBeta(gs, nextMoves, depth - 1 - reduction,
                                                   -alpha - 1, -alpha, -turnMultiplier,
                                                   root_depth=root_depth,
                                                   position_history=position_history)
                if score > alpha:
                    score = -findMoveNegaMaxAlphaBeta(gs, nextMoves, depth - 1,
                                                       -beta, -alpha, -turnMultiplier,
                                                       root_depth=root_depth,
                                                       position_history=position_history)
            else:
                score = -findMoveNegaMaxAlphaBeta(gs, nextMoves, depth - 1,
                                                   -beta, -alpha, -turnMultiplier,
                                                   root_depth=root_depth,
                                                   position_history=position_history)

        gs.undoMove()

        if score > maxScore:
            maxScore = score
            best_move_this_node = move
            if depth == root_depth:
                nextMove = move
                past_moves[score] = move

        if maxScore > alpha:
            alpha = maxScore
            if move.pieceCaptured == "--" and not move.pawnPromotion:
                h_key = (move.pieceMoved, move.endRow, move.endCol)
                history_table[h_key] = history_table.get(h_key, 0) + depth * depth

        if alpha >= beta:
            if move.pieceCaptured == "--" and not move.pawnPromotion:
                killers = killer_moves.setdefault(depth, [])
                if move not in killers:
                    killers.insert(0, move)
                    if len(killers) > 2:
                        killers.pop()
            break

    # ── Undo the position visit after searching ──
    position_history[zobrist_hash] -= 1

    # ── Store result in TT with flag and best move (4-tuple) ──
    if maxScore <= original_alpha:
        tt_flag = 'UPPER'
    elif maxScore >= beta:
        tt_flag = 'LOWER'
    else:
        tt_flag = 'EXACT'
    transposition_table[zobrist_hash] = (maxScore, depth, tt_flag, best_move_this_node)

    return maxScore - repetition_penalty


def scoreMaterial(board):
    score = 0
    count = 0
    for row in board:
        for square in row:
            if square[0] == 'w':
                score += pieceScore[square[1]]
                count += 1
            elif square[0] == 'b':
                score += pieceScore[square[1]]
                count += 1
    return score, count


def board_to_fen(board, flip=False):
    """
    Converts an 8x8 board where squares are like "wP","bq" or "--" into the FEN piece-placement string.
    """
    fen_parts = []
    rows = board if not flip else board[::-1]
    for row in rows:
        empty_count = 0
        fen_row = ""
        for piece in row:
            if piece == "--" or piece is None or piece == "":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                side = piece[0].lower()
                symbol = piece[1].lower()
                if side == 'w':
                    fen_row += symbol.upper()
                else:
                    fen_row += symbol
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_parts.append(fen_row)
    return "/".join(fen_parts)


# ═══════════════════════════════════════════════════════════════════════
#  POSITIONAL WEIGHT SYSTEM  (linear bias layered on top of NNUE)
#
#  Formula:
#      final_score = nnue_score * 1.0  +  clamp(bias, -0.30, +0.30)
#
#  Like linear regression:  y = x*w_nnue + features*w_pos
#  w_nnue is fixed at 1.0 so NNUE always dominates.
#  The positional nudge is capped at ±0.30 pawns — never overrides NNUE.
#
#  Two weighted features:
#    w1 = 0.35  — Piece-square tables, phase-blended opening ↔ endgame
#    w2 = 0.50 × (1 − phase_t)  — Development bonus, fades to 0 in endgame
# ═══════════════════════════════════════════════════════════════════════

# ── Piece-square tables (White's perspective, row 0 = rank 8) ─────────
# Values in pawn units. Black mirrors vertically via (7 - row).

PST_PAWN_OPENING = [
    [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
    [ 0.05,  0.10,  0.10, -0.20, -0.20,  0.10,  0.10,  0.05],
    [ 0.05, -0.05, -0.10,  0.00,  0.00, -0.10, -0.05,  0.05],
    [ 0.00,  0.00,  0.00,  0.20,  0.20,  0.00,  0.00,  0.00],
    [ 0.05,  0.05,  0.10,  0.25,  0.25,  0.10,  0.05,  0.05],
    [ 0.10,  0.10,  0.20,  0.30,  0.30,  0.20,  0.10,  0.10],
    [ 0.50,  0.50,  0.50,  0.50,  0.50,  0.50,  0.50,  0.50],
    [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
]

PST_PAWN_ENDGAME = [
    [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
    [-0.05, -0.05,  0.00,  0.00,  0.00,  0.00, -0.05, -0.05],
    [-0.05, -0.05,  0.00,  0.05,  0.05,  0.00, -0.05, -0.05],
    [ 0.00,  0.00,  0.10,  0.20,  0.20,  0.10,  0.00,  0.00],
    [ 0.10,  0.10,  0.20,  0.30,  0.30,  0.20,  0.10,  0.10],
    [ 0.25,  0.25,  0.30,  0.35,  0.35,  0.30,  0.25,  0.25],
    [ 0.50,  0.50,  0.50,  0.50,  0.50,  0.50,  0.50,  0.50],
    [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
]

PST_KNIGHT = [
    [-0.50, -0.40, -0.30, -0.30, -0.30, -0.30, -0.40, -0.50],
    [-0.40, -0.20,  0.00,  0.05,  0.05,  0.00, -0.20, -0.40],
    [-0.30,  0.05,  0.10,  0.15,  0.15,  0.10,  0.05, -0.30],
    [-0.30,  0.00,  0.15,  0.20,  0.20,  0.15,  0.00, -0.30],
    [-0.30,  0.05,  0.15,  0.20,  0.20,  0.15,  0.05, -0.30],
    [-0.30,  0.00,  0.10,  0.15,  0.15,  0.10,  0.00, -0.30],
    [-0.40, -0.20,  0.00,  0.00,  0.00,  0.00, -0.20, -0.40],
    [-0.50, -0.40, -0.30, -0.30, -0.30, -0.30, -0.40, -0.50],
]

PST_BISHOP = [
    [-0.20, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.20],
    [-0.10,  0.05,  0.00,  0.00,  0.00,  0.00,  0.05, -0.10],
    [-0.10,  0.10,  0.10,  0.10,  0.10,  0.10,  0.10, -0.10],
    [-0.10,  0.00,  0.10,  0.10,  0.10,  0.10,  0.00, -0.10],
    [-0.10,  0.05,  0.05,  0.10,  0.10,  0.05,  0.05, -0.10],
    [-0.10,  0.00,  0.05,  0.10,  0.10,  0.05,  0.00, -0.10],
    [-0.10,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.10],
    [-0.20, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.20],
]

PST_ROOK = [
    [ 0.00,  0.00,  0.00,  0.05,  0.05,  0.00,  0.00,  0.00],
    [-0.05,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.05],
    [-0.05,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.05],
    [-0.05,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.05],
    [-0.05,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.05],
    [-0.05,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.05],
    [ 0.05,  0.10,  0.10,  0.10,  0.10,  0.10,  0.10,  0.05],
    [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
]

PST_QUEEN = [
    [-0.20, -0.10, -0.10, -0.05, -0.05, -0.10, -0.10, -0.20],
    [-0.10,  0.00,  0.05,  0.00,  0.00,  0.00,  0.00, -0.10],
    [-0.10,  0.05,  0.05,  0.05,  0.05,  0.05,  0.00, -0.10],
    [ 0.00,  0.00,  0.05,  0.05,  0.05,  0.05,  0.00, -0.05],
    [-0.05,  0.00,  0.05,  0.05,  0.05,  0.05,  0.00, -0.05],
    [-0.10,  0.00,  0.05,  0.05,  0.05,  0.05,  0.00, -0.10],
    [-0.10,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.10],
    [-0.20, -0.10, -0.10, -0.05, -0.05, -0.10, -0.10, -0.20],
]

PST_KING_OPENING = [
    [ 0.20,  0.30,  0.10,  0.00,  0.00,  0.10,  0.30,  0.20],
    [ 0.20,  0.20,  0.00,  0.00,  0.00,  0.00,  0.20,  0.20],
    [-0.10, -0.20, -0.20, -0.20, -0.20, -0.20, -0.20, -0.10],
    [-0.20, -0.30, -0.30, -0.40, -0.40, -0.30, -0.30, -0.20],
    [-0.30, -0.40, -0.40, -0.50, -0.50, -0.40, -0.40, -0.30],
    [-0.30, -0.40, -0.40, -0.50, -0.50, -0.40, -0.40, -0.30],
    [-0.30, -0.40, -0.40, -0.50, -0.50, -0.40, -0.40, -0.30],
    [-0.30, -0.40, -0.40, -0.50, -0.50, -0.40, -0.40, -0.30],
]

PST_KING_ENDGAME = [
    [-0.50, -0.30, -0.30, -0.30, -0.30, -0.30, -0.30, -0.50],
    [-0.30, -0.30,  0.00,  0.00,  0.00,  0.00, -0.30, -0.30],
    [-0.30, -0.10,  0.20,  0.30,  0.30,  0.20, -0.10, -0.30],
    [-0.30, -0.10,  0.30,  0.40,  0.40,  0.30, -0.10, -0.30],
    [-0.30, -0.10,  0.30,  0.40,  0.40,  0.30, -0.10, -0.30],
    [-0.30, -0.10,  0.20,  0.30,  0.30,  0.20, -0.10, -0.30],
    [-0.30, -0.20, -0.10,  0.00,  0.00, -0.10, -0.20, -0.30],
    [-0.50, -0.40, -0.30, -0.20, -0.20, -0.30, -0.40, -0.50],
]


def get_game_phase(board):
    """
    Returns (phase_t, is_opening):
      phase_t   : float  0.0 = opening/early-mid,  1.0 = deep endgame
      is_opening: True when minor pieces are still undeveloped on back ranks

    phase_t is computed from remaining non-pawn material.
    24 = full set (Q×1 + R×2 + B×2 + N×2) × 2 sides, weighted by phase weights.
    """
    _phase_weights = {"Q": 4, "R": 2, "B": 1, "N": 1}
    npm = 0          # non-pawn material remaining (weighted)
    undeveloped = 0  # minor pieces still on starting rows

    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if p == "--":
                continue
            piece, side = p[1], p[0]
            npm += _phase_weights.get(piece, 0)
            if piece in ("N", "B"):
                if (side == 'w' and r == 7) or (side == 'b' and r == 0):
                    undeveloped += 1

    # npm goes from ~24 (full board) down to 0 (bare kings)
    phase_t = 1.0 - min(npm / 24.0, 1.0)
    is_opening = undeveloped >= 2
    return phase_t, is_opening


def get_pst_value(piece, side, r, c, phase_t):
    """
    Look up piece-square table value for one piece.
    Always returns a value from White's perspective
    (positive = good for White, negative = good for Black).

    Black pieces mirror the table vertically: row = 7 - r.
    Pawn and King tables are interpolated between opening and endgame
    using phase_t (0 = full opening table, 1 = full endgame table).
    """
    row = r if side == 'w' else (7 - r)

    if piece == 'P':
        val = (PST_PAWN_OPENING[row][c] * (1.0 - phase_t)
               + PST_PAWN_ENDGAME[row][c] * phase_t)
    elif piece == 'N':
        val = PST_KNIGHT[row][c]
    elif piece == 'B':
        val = PST_BISHOP[row][c]
    elif piece == 'R':
        val = PST_ROOK[row][c]
    elif piece == 'Q':
        val = PST_QUEEN[row][c]
    elif piece == 'K':
        val = (PST_KING_OPENING[row][c] * (1.0 - phase_t)
               + PST_KING_ENDGAME[row][c] * phase_t)
    else:
        val = 0.0

    # Flip sign for Black: what is good for Black is bad for White
    return val if side == 'w' else -val


def get_development_bonus(board, is_opening, phase_t):
    if not is_opening:
        return 0.0
    bonus = 0.0
    extended_center = {(3, 3), (3, 4), (4, 3), (4, 4),
                       (2, 3), (2, 4), (5, 3), (5, 4)}
    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if p == "--":
                continue
            piece, side = p[1], p[0]
            sign = 1 if side == 'w' else -1
            # Reward minor pieces that have left the back rank
            if piece in ("N", "B"):
                if (side == 'w' and r < 7) or (side == 'b' and r > 0):
                    bonus += sign * 0.08
            # Small penalty for early queen development
            # (discourages Qh5 on move 2 etc.)
            if piece == 'Q':
                if (side == 'w' and r < 7) or (side == 'b' and r > 0):
                    bonus -= sign * 0.04
            # Center control reward for pawns and minor pieces
            if (r, c) in extended_center and piece in ("P", "N", "B"):
                bonus += sign * 0.05
    return bonus


def get_positional_bias(board):
    """
    Master function — computes the total positional bias to add on top of NNUE.
    Formula (matches linear regression structure):
        bias = pst_score * w1  +  dev_score * w2(phase_t)
    Weights:
        w1 = 0.35   — PST weight, constant across all phases
        w2 = 0.50 × (1 − phase_t)   — development weight, zero in endgame
    The result is hard-clamped to ±0.30 pawns so that NNUE always
    dominates and a PST quirk can never cause a blunder.
    NNUE typically returns ±2–8 pawns, so ±0.30 is a 4–15% nudge at most.
    """
    phase_t, is_opening = get_game_phase(board)

    # ── w1: Piece-square table score (always active) ──────────────────
    w1 = 0.35
    pst_score = 0.0
    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if p != "--":
                pst_score += get_pst_value(p[1], p[0], r, c, phase_t)

    # ── w2: Development bonus (opening only, fades with phase) ────────
    w2 = 0.50 * (1.0 - phase_t)
    dev_score = get_development_bonus(board, is_opening, phase_t)

    raw_bias = pst_score * w1 + dev_score * w2

    # ── Hard clamp: NNUE stays in charge ─────────────────────────────
    return max(-0.30, min(0.30, raw_bias))


def get_mopup_score(board):
    """
    Mopup evaluation for winning endgames (KQ vs K, KR vs K, etc.).

    When one side is up by >= 4 pawns of material (a rook or more), the engine
    receives two incentives that guide king-and-piece coordination:
      1. Push the losing king toward a corner (corner_dist bonus).
      2. Bring the winning king closer to the losing king (proximity bonus).

    Without this, a depth-2 search has zero gradient toward mate: every king
    shuffle scores identically via NNUE because NNUE sees the material but
    cannot "feel" positional urgency at only 2 ply.

    Returns score in pawn units from White's perspective.
    Positive  = White is the winning side and mopup is progressing.
    Negative  = Black is the winning side and mopup is progressing.
    """
    piece_vals = {"Q": 9, "R": 5, "B": 3, "N": 3, "P": 1, "K": 0}
    white_material = 0
    black_material = 0
    white_king_pos = None
    black_king_pos = None
    total_pieces = 0

    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if p == "--":
                continue
            total_pieces += 1
            if p[1] == 'K':
                if p[0] == 'w':
                    white_king_pos = (r, c)
                else:
                    black_king_pos = (r, c)
            else:
                if p[0] == 'w':
                    white_material += piece_vals.get(p[1], 0)
                else:
                    black_material += piece_vals.get(p[1], 0)

    if white_king_pos is None or black_king_pos is None:
        return 0.0

    material_diff = white_material - black_material

    # Guard 1: Only activate in genuine endgames (<=10 pieces total).
    # During middlegame, quiescence leaf nodes can temporarily show
    # material_diff >= 5 (e.g. a rook captured in an unresolved exchange).
    # Without this guard the mopup adds 1-2 pawn king-proximity bonus at
    # those nodes, distorting evaluations and causing middlegame blunders.
    if total_pieces > 10:
        return 0.0

    # Guard 2: Only kick in when up by at least a rook (5 pawns).
    if abs(material_diff) < 5:
        return 0.0

    if material_diff > 0:
        winning_king = white_king_pos
        losing_king  = black_king_pos
        sign = 1
    else:
        winning_king = black_king_pos
        losing_king  = white_king_pos
        sign = -1

    lr, lc = losing_king
    wr, wc = winning_king

    # Corner distance: losing king's Manhattan distance from center 3-4 zone.
    # 0 = centre, 6 = corner — the engine is rewarded for increasing this.
    corner_dist = max(3 - lr, lr - 4, 0) + max(3 - lc, lc - 4, 0)

    # King proximity: Manhattan distance between kings (1–14).
    # Rewarded for DECREASING this (bring kings together for mating net).
    king_dist = abs(lr - wr) + abs(lc - wc)

    # Scale chosen so total mopup ≤ ~2 pawns — significant enough to guide
    # the king but never large enough to override the 5-9 pawn material edge.
    mopup = 0.15 * corner_dist + 0.08 * (14 - king_dist)

    return sign * mopup


def scoreBoard(gs):
    """
    Evaluates the current board position and returns a score in pawn units.

    Positive  = White is better
    Negative  = Black is better
    ±CHECKMATE = forced checkmate detected

    Pipeline:
        1. Convert board → FEN string
        2. Check terminal conditions (checkmate / draw) via python-chess
        3. Call NNUE C DLL → raw centipawn score → divide by 200 → pawn units
        4. Add positional bias (PST + development, clamped ±0.30)

    The positional bias encourages human-like opening play (piece development,
    centre control, king safety) without overriding NNUE on tactical positions.
    """
    fen = board_to_fen(gs.board)
    fen_full = f"{fen} {'w' if gs.whiteToMove else 'b'} - - 0 1"

    chess_board = chess.Board(fen_full)
    if chess_board.is_checkmate():
        return -CHECKMATE if gs.whiteToMove else CHECKMATE
    elif chess_board.is_stalemate() or chess_board.is_insufficient_material() or \
         chess_board.is_fivefold_repetition() or chess_board.is_seventyfive_moves():
        return STALEMATE

    # ── Base evaluation ─────────────────────────────────────────────────
    if nnue is not None:
        # NNUE evaluation (pawn units)
        nnue_score = nnue.nnue_evaluate_fen(fen_full.encode("utf-8")) / 200
    else:
        # Fallback: material count + PST (no NNUE library available)
        piece_vals = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}
        nnue_score = 0
        for r in range(8):
            for c in range(8):
                p = gs.board[r][c]
                if p != "--":
                    val = piece_vals.get(p[1], 0)
                    nnue_score += val if p[0] == 'w' else -val

    # ── Positional bias: y = x*w_nnue + features*w_pos ───────────────
    positional_bias = get_positional_bias(gs.board)

    # ── Mopup: guides king activity in winning endgames ───────────────
    mopup = get_mopup_score(gs.board)

    return nnue_score + positional_bias + mopup