import sys
import os
import pytest
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app, games

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        with app.app_context():
            yield client

def test_threefold_repetition_with_undo(client):
    # Start a new game
    resp = client.post("/new_game", json={"mode": "ai", "color": "white", "engine": "negamax_2"})
    data = resp.get_json()
    gid = None
    with client.session_transaction() as sess:
        gid = sess["game_id"]
    
    # Enable human-to-human for testing
    game = games[gid]
    game["player_white"] = True
    game["player_black"] = True
    game["mode"] = "human"
    
    # Start position hash
    start_hash = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
    assert game["position_history"][start_hash] == 1
    
    # Move Nf3
    client.post("/human_move", json={"move": "g1f3"}) # w
    client.post("/human_move", json={"move": "g8f6"}) # b
    client.post("/human_move", json={"move": "f3g1"}) # w
    client.post("/human_move", json={"move": "f6g8"}) # b (2nd time)
    
    assert game["position_history"][start_hash] == 2
    
    client.post("/human_move", json={"move": "g1f3"}) # w
    client.post("/human_move", json={"move": "g8f6"}) # b
    client.post("/human_move", json={"move": "f3g1"}) # w
    resp = client.post("/human_move", json={"move": "f6g8"}) # b (3rd time)
    data = resp.get_json()
    
    assert game["position_history"][start_hash] == 3
    assert data["msg"] == "Draw by threefold repetition"
    
    # Undo
    client.post("/undo")
    assert game["position_history"][start_hash] == 2
    assert len(game["move_history"]) == 7 # f6g8 was popped
