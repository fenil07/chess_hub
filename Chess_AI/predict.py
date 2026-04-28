import os
import numpy as np
import torch
import torch.nn as nn
import pickle
import logging

# --- ABSOLUTE PATH CONFIGURATION ---
base_dir = os.path.dirname(__file__)
CHECKPOINT_PATH = os.path.join(base_dir, "model", "CHESS_MODEL.pth")
MAPPING_FILE = os.path.join(base_dir, "model", "move_to_int")


# --- Model Architecture ---
# Must exactly match the layer names used during training so the state_dict keys align!
class ChessModel(nn.Module):
    def __init__(self, num_classes):
        super(ChessModel, self).__init__()
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output raw logits
        return x

    # 💡 THE FIX: Mimic TensorFlow's .predict() method!
    def predict(self, x, verbose=0):
        self.eval()
        with torch.no_grad():
            # 'x' comes in as a numpy array from your app's np.expand_dims
            tensor_x = torch.tensor(x, dtype=torch.float32).to(next(self.parameters()).device)
            logits = self.forward(tensor_x)
            # Return a numpy array so tf.nn.softmax can handle it in your app
            return logits.cpu().numpy()


class ChessPredictor:
    def __init__(self):
        if not os.path.exists(MAPPING_FILE):
            raise FileNotFoundError(f"[ERROR] Mapping file not found! Expected at: {MAPPING_FILE}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.move_to_int = self._load_mapping()
        self.int_to_move = {v: k for k, v in self.move_to_int.items()}

        self.num_classes = len(self.move_to_int)
        logging.debug(f"[OK] Num classes: {self.num_classes}")
        self.model = self._build_and_load_model()

    def _load_mapping(self):
        with open(MAPPING_FILE, 'rb') as f:
            return pickle.load(f)

    def _build_and_load_model(self):
        model = ChessModel(self.num_classes).to(self.device)

        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            logging.debug(f"[OK] PyTorch Model Loaded from: {CHECKPOINT_PATH}")
        else:
            logging.warning(f"[WARN] Warning: Checkpoint not found at {CHECKPOINT_PATH}. Using random weights.")

        return model

    def gs_to_matrix(self, gs):
        matrix = np.zeros((13, 8, 8), dtype=np.float32)
        piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}

        for r in range(8):
            for c in range(8):
                piece = gs.board[r][c]
                if piece != "--":
                    piece_type = piece_map[piece[1]]
                    piece_color = 0 if piece[0] == 'w' else 6
                    chess_row = 7 - r
                    matrix[piece_type + piece_color, chess_row, c] = 1

        valid_moves = gs.getValidMoves()
        for m in valid_moves:
            chess_row_to = 7 - m.endRow
            matrix[12, chess_row_to, m.endCol] = 1

        return matrix

    def get_prediction(self, gs):
        matrix = self.gs_to_matrix(gs)
        X_input = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(X_input)
            # Use float64 for softmax to avoid underflow with large negative logits
            probabilities = torch.softmax(logits.double(), dim=1).cpu().float().numpy()[0]
        return probabilities


# Initialize globally
predictor = ChessPredictor()