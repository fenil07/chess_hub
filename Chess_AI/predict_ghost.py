import os
import numpy as np
import torch
import torch.nn as nn
import json
import logging

# ---------------------------------------------------------------------------
#  ABSOLUTE PATH CONFIGURATION
#  Update these two paths to match your local file structure.
# ---------------------------------------------------------------------------
base_dir = os.path.dirname(__file__)
CHECKPOINT_PATH = os.path.join(base_dir, "model", "ghost_chess_best.pth")
MAPPING_FILE    = os.path.join(base_dir, "model", "move_to_int.json")


# ---------------------------------------------------------------------------
#  MODEL ARCHITECTURE  —  GhostChessNet
#  Must exactly mirror the class definitions in model_pytorch_2.py so that
#  the saved state_dict keys align when we call model.load_state_dict().
# ---------------------------------------------------------------------------

class GhostModule(nn.Module):
    """Efficient Ghost Convolution: one primary conv + one cheap depthwise conv."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        half = out_ch // 2
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, half, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(half, half, 3, padding=1, groups=half, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        p = self.primary(x)
        return torch.cat([p, self.cheap(p)], dim=1)


class ChessAttention(nn.Module):
    """Spatial + Channel attention gate fused into a single module."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.spatial_conv = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.channel_fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x):
        B, C, H, W_ = x.shape
        s_gate = torch.sigmoid(self.spatial_conv(x.mean(dim=1, keepdim=True)))
        c_gate = torch.sigmoid(self.channel_fc(x.mean(dim=[2, 3])).view(B, C, 1, 1))
        return x * s_gate * c_gate


def extract_phase_features(x: torch.Tensor) -> torch.Tensor:
    """
    Derive 4 scalar game-phase features from the 13-plane board tensor.
    These are concatenated with the flattened CNN output before the head.
    """
    piece_planes      = x[:, :12]
    pawn_planes       = x[:, [0, 6]]
    legal_plane       = x[:, 12]
    material_density  = piece_planes.sum(dim=[1, 2, 3]) / (12 * 64.0)
    pawn_density      = pawn_planes.sum(dim=[1, 2, 3])  / (2  * 64.0)
    piece_mobility    = legal_plane.sum(dim=[1, 2])      / 64.0
    major             = x[:, [3, 4, 9, 10]].sum(dim=[1, 2, 3])
    total_pieces      = piece_planes.sum(dim=[1, 2, 3]).clamp(min=1.0)
    major_piece_ratio = (major / total_pieces).clamp(0.0, 1.0)
    return torch.stack(
        [material_density, pawn_density, piece_mobility, major_piece_ratio], dim=1
    )


class GhostChessNet(nn.Module):
    """
    Ghost-module CNN for chess move prediction.

    Input  : (batch, 13, 8, 8)  — 12 piece planes + 1 legal-move plane
    Output : (batch, num_classes) raw logits — one per legal UCI move string

    Architecture:
        stem  → ghost1 (64→128) → ghost2 (128→256) → attention
        → AdaptiveAvgPool(4×4) → flatten (4096)
        → concat phase features (4) → total 4100
        → decision_head (4100 → 512 → 256 → num_classes)
    """
    def __init__(self, num_classes: int):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(13, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.ghost1    = GhostModule(64, 128)
        self.ghost2    = GhostModule(128, 256)
        self.attention = ChessAttention(256, reduction=8)

        self.pool    = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()

        # 256 channels × 4×4 spatial = 4096, + 4 phase features = 4100
        self.decision_head = nn.Sequential(
            nn.Linear(4100, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.01),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.01),

            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = extract_phase_features(x)
        out   = self.stem(x)
        out   = self.ghost1(out)
        out   = self.ghost2(out)
        out   = self.attention(out)
        out   = self.pool(out)
        out   = self.flatten(out)
        out   = torch.cat([out, phase], dim=1)   # (batch, 4100)
        return self.decision_head(out)


# ---------------------------------------------------------------------------
#  PREDICTOR CLASS
# ---------------------------------------------------------------------------

class GhostPredictor:
    """
    Wraps GhostChessNet for inference inside the Flask app.

    Usage:
        ghost_predictor.get_prediction(gs)  →  np.ndarray of shape (num_classes,)
    """

    def __init__(self):
        if not os.path.exists(MAPPING_FILE):
            raise FileNotFoundError(
                f"[ERROR] Ghost move-mapping not found! Expected at: {MAPPING_FILE}\n"
                f"   Make sure move_to_int.json was generated by model_pytorch_2.py training."
            )

        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.move_to_int = self._load_mapping()
        self.int_to_move = {v: k for k, v in self.move_to_int.items()}
        self.num_classes = len(self.move_to_int)

        logging.debug(f"[OK] Ghost Predictor -- {self.num_classes:,} move classes  |  device: {self.device}")
        self.model = self._build_and_load_model()

    # ── Private helpers ──────────────────────────────────────────────────

    def _load_mapping(self) -> dict:
        with open(MAPPING_FILE, 'r') as f:
            return json.load(f)

    def _build_and_load_model(self) -> GhostChessNet:
        model = GhostChessNet(self.num_classes).to(self.device)

        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(
                CHECKPOINT_PATH, map_location=self.device, weights_only=False
            )
            # Support both raw state_dict and the training wrapper dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            logging.debug(f"[OK] GhostChessNet loaded from: {CHECKPOINT_PATH}")
        else:
            model.eval()
            logging.warning(
                f"[WARN] Checkpoint not found at {CHECKPOINT_PATH}. "
                f"Ghost AI will play with random weights until the model file is present."
            )

        return model

    # ── Board encoding ───────────────────────────────────────────────────

    def gs_to_matrix(self, gs) -> np.ndarray:
        """
        Convert a ChessEngine.GameState object into a (13, 8, 8) float32 tensor.

        Planes 0–5  : white pieces  (P, N, B, R, Q, K)
        Planes 6–11 : black pieces  (p, n, b, r, q, k)
        Plane  12   : legal-move destination squares for side to move
        """
        matrix   = np.zeros((13, 8, 8), dtype=np.float32)
        piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}

        for r in range(8):
            for c in range(8):
                piece = gs.board[r][c]
                if piece != "--":
                    piece_type  = piece_map[piece[1]]
                    piece_color = 0 if piece[0] == 'w' else 6
                    chess_row   = 7 - r          # flip: rank 1 → row index 0
                    matrix[piece_type + piece_color, chess_row, c] = 1

        # Legal-move plane: mark destination squares of all valid moves
        valid_moves = gs.getValidMoves()
        for m in valid_moves:
            chess_row_to = 7 - m.endRow
            matrix[12, chess_row_to, m.endCol] = 1

        return matrix

    # ── Inference ────────────────────────────────────────────────────────

    def get_prediction(self, gs) -> np.ndarray:
        """
        Return a probability distribution over all move classes.

        Args:
            gs : ChessEngine.GameState — current board position

        Returns:
            np.ndarray, shape (num_classes,), dtype float32
            Values sum to ~1 (softmax output).
        """
        matrix  = self.gs_to_matrix(gs)
        X_input = (
            torch.tensor(matrix, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            logits        = self.model(X_input)
            # Use float64 for softmax to avoid underflow with large logit differences
            probabilities = (
                torch.softmax(logits.double(), dim=1)
                .cpu()
                .float()
                .numpy()[0]
            )
        return probabilities


# ---------------------------------------------------------------------------
#  GLOBAL SINGLETON  (imported by ChessAI.py)
# ---------------------------------------------------------------------------
ghost_predictor = GhostPredictor()