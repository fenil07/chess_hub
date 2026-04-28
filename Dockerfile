FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860
RUN apt-get update && apt-get install -y wget tar make gcc g++ libstdc++6 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Models and Data from Dataset
RUN mkdir -p Chess_AI/RF_model Chess_AI/model Chess_AI/src && \
    wget -qO Chess_AI/RF_model/opening_rf.pkl https://huggingface.co/datasets/Fenil045/chess_models/resolve/main/opening_rf.pkl && \
    wget -qO Chess_AI/src/puzzles.duckdb https://huggingface.co/datasets/Fenil045/chess_models/resolve/main/puzzles.duckdb && \
    wget -qO Chess_AI/model/CHESS_MODEL.pth https://huggingface.co/datasets/Fenil045/chess_models/resolve/main/CHESS_MODEL.pth && \
    wget -qO Chess_AI/model/ghost_chess_best.pth https://huggingface.co/datasets/Fenil045/chess_models/resolve/main/ghost_chess_best.pth && \
    wget -qO Chess_AI/model/move_to_int https://huggingface.co/datasets/Fenil045/chess_models/resolve/main/move_to_int

# FIXED: Updated Stockfish URL to .tar (Stockfish 18 format)
RUN mkdir -p stockfish && \
    wget -qO stockfish.tar https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar && \
    tar -xf stockfish.tar && \
    find . -name "stockfish-ubuntu-x86-64-avx2" -type f -exec mv {} stockfish/stockfish \; && \
    chmod +x stockfish/stockfish && \
    rm -rf stockfish.tar stockfish-ubuntu-*

COPY --chown=user . .
RUN cd Chess_AI/src && make clean 2>/dev/null || true && make COMP=gcc
EXPOSE 7860
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1"]
