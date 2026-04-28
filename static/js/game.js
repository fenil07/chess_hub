window.addEventListener('popstate', function(e) {
    if (window._gameOver) return;
    history.pushState(null, '', window.location.href);
    if(typeof showResignModal === 'function') showResignModal();
});
history.pushState(null, '', window.location.href);

// ── State ─────────────────────────────────────────────────────────────
let board              = null;
let selectedSquare     = null;
let validMovesForPiece = [];
let moveCount          = 1;
let isPlayerTurn       = true;
let gameMode           = 'ai';
let playerColor        = 'white';
let aiEngine           = 'model';   // 'model' | 'negamax_2' | 'negamax_3'
let currentTurnColor   = 'white';
let gameOver           = false;
let pendingPromo       = null;  // { uciBase: 'e7e8', color: 'w' } when awaiting promotion choice

// ── Move history for board replay (#14) ───────────────────────────────
let fenHistory  = [];
let replayMode  = false;
let replayIndex = -1;

// ── Chess Clock (#16) ─────────────────────────────────────────────────
let clockEnabled  = false;
let timeWhite     = 0;
let timeBlack     = 0;
let clockInterval = null;
let clockRunning  = false;

// ── Boot ──────────────────────────────────────────────────────────────
function startNewGame() {
    const urlParams = new URLSearchParams(window.location.search);
    gameMode    = urlParams.get('mode')   || 'ai';
    playerColor = urlParams.get('color')  || 'white';
    aiEngine    = urlParams.get('engine') || 'model';

    const timeControl = parseInt(urlParams.get('time') || '0');
    clockEnabled = timeControl > 0;
    if (clockEnabled) {
        timeWhite = timeControl * 60;
        timeBlack = timeControl * 60;
        renderClocks();
        document.getElementById('clock-section').style.display = 'flex';
    }

    return fetch('/new_game', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ mode: gameMode, color: playerColor, engine: aiEngine })
    })
    .then(r => r.json())
    .then(data => {
        board = Chessboard('board', {
            position:    'start',
            draggable:   false,
            orientation: playerColor,
            moveSpeed:   550,
            pieceTheme:  '/Chess_AI/images/{piece}.png'
        });

        board.position(data.fen, false);

        const whiteEl = document.getElementById('white-name');
        const blackEl = document.getElementById('black-name');
        whiteEl.innerText    = data.w_name;
        blackEl.innerText    = data.b_name;
        whiteEl.dataset.name = data.w_name;
        blackEl.dataset.name = data.b_name;
        whiteEl.dataset.isAi = (playerColor === 'black') ? 'true' : 'false';
        blackEl.dataset.isAi = (playerColor === 'white') ? 'true' : 'false';

        fenHistory       = [data.fen];
        replayMode       = false;
        replayIndex      = -1;
        isPlayerTurn     = true;
        currentTurnColor = 'white';
        gameOver         = false;
        moveCount        = 1;
        document.getElementById('move-log').innerHTML = '';

        if (data.ai_san) {
            fenHistory.push(data.fen);
            addMoveToLog(data.ai_san, 'ai', data.fen);
            currentTurnColor = 'black';
        }

        if (clockEnabled) startClock(currentTurnColor);
    })
    .catch(err => console.error('new_game error:', err));
}

// ── Chess Clock ───────────────────────────────────────────────────────
function renderClocks() {
    document.getElementById('clock-white').textContent = formatTime(timeWhite);
    document.getElementById('clock-black').textContent = formatTime(timeBlack);
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
}

function startClock(color) {
    if (!clockEnabled) return;
    stopClock();
    clockRunning = true;
    document.getElementById('clock-white').classList.toggle('clock-active', color === 'white');
    document.getElementById('clock-black').classList.toggle('clock-active', color === 'black');
    clockInterval = setInterval(() => {
        if (color === 'white') {
            timeWhite = Math.max(0, timeWhite - 1);
            document.getElementById('clock-white').textContent = formatTime(timeWhite);
            if (timeWhite <= 10) document.getElementById('clock-white').classList.add('clock-low');
            if (timeWhite === 0) { stopClock(); handleTimeout('white'); }
        } else {
            timeBlack = Math.max(0, timeBlack - 1);
            document.getElementById('clock-black').textContent = formatTime(timeBlack);
            if (timeBlack <= 10) document.getElementById('clock-black').classList.add('clock-low');
            if (timeBlack === 0) { stopClock(); handleTimeout('black'); }
        }
    }, 1000);
}

function stopClock() {
    if (clockInterval) { clearInterval(clockInterval); clockInterval = null; }
    clockRunning = false;
    document.getElementById('clock-white').classList.remove('clock-active');
    document.getElementById('clock-black').classList.remove('clock-active');
}

function handleTimeout(color) {
    if (gameOver) return;
    gameOver = true; isPlayerTurn = false;
    showResultModal('Time Out', (color === 'white' ? 'Black' : 'White') + ' wins on time!');
}

// ── Square click ──────────────────────────────────────────────────────
function onSquareClick(square) {
    if (replayMode) { exitReplayMode(); return; }
    if (!isPlayerTurn || gameOver) return;

    if (selectedSquare === square) { deselectPiece(); return; }

    if (selectedSquare) {
        const base4 = selectedSquare + square;

        // Check if ANY valid move from this source→target is a promotion
        const isPromoMove = validMovesForPiece.some(
            m => m.length === 5 && m.slice(0, 4) === base4
        );

        if (isPromoMove) {
            // Show promotion picker — the user will pick the piece
            // Use currentTurnColor (not playerColor) so pass-and-play Black promotion shows black pieces
            const pieceColor = currentTurnColor === 'white' ? 'w' : 'b';
            pendingPromo = { uciBase: base4, color: pieceColor, targetSquare: square };
            deselectPiece();
            showInlinePromotion(pieceColor, square);
            return;
        }

        const uci = base4;
        if (validMovesForPiece.includes(uci)) { executeMove(uci); return; }
        deselectPiece();
    }

    selectedSquare = square;
    fetch('/get_valid_moves', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ square })
    })
    .then(r => r.json())
    .then(data => {
        if (data.moves && data.moves.length > 0) {
            validMovesForPiece = data.moves;
            highlightSquare(square, 'selected-square');
            // Deduplicate highlight squares (4 promotion moves go to the same square)
            const uniqueTargets = [...new Set(data.moves.map(m => m.slice(2, 4)))];
            uniqueTargets.forEach(sq => highlightSquare(sq, 'legal-highlight'));
        } else { selectedSquare = null; }
    })
    .catch(() => { selectedSquare = null; });
}

// ── CSS-based piece animation ─────────────────────────────────────────
function animatePieceMove(fromSq, toSq, durationMs, callback) {
    const fromCell = document.querySelector('#board .square-' + fromSq);
    const toCell   = document.querySelector('#board .square-' + toSq);
    const pieceImg = fromCell ? fromCell.querySelector('img') : null;
    if (!pieceImg || !toCell) { callback(); return; }

    const dx = toCell.getBoundingClientRect().left - fromCell.getBoundingClientRect().left;
    const dy = toCell.getBoundingClientRect().top  - fromCell.getBoundingClientRect().top;
    pieceImg.style.position   = 'relative';
    pieceImg.style.zIndex     = '9999';
    pieceImg.style.transition = 'transform ' + durationMs + 'ms cubic-bezier(0.25,0.46,0.45,0.94)';
    pieceImg.getBoundingClientRect();
    pieceImg.style.transform  = 'translate(' + dx + 'px,' + dy + 'px)';
    setTimeout(() => {
        pieceImg.style.transition = '';
        pieceImg.style.transform  = '';
        pieceImg.style.position   = '';
        pieceImg.style.zIndex     = '';
        callback();
    }, durationMs);
}

function executeMove(uci) {
    deselectPiece();
    isPlayerTurn = false;
    if (clockEnabled) stopClock();

    var from = uci.slice(0, 2), to = uci.slice(2, 4), ANIM = 380;
    var animDone = false, serverData = null;

    function tryFinalize() {
        if (!animDone || !serverData) return;
        if (serverData.error) { board.position(serverData.fen || '', false); isPlayerTurn = true; return; }

        board.position(serverData.fen_after_human, false);
        fenHistory.push(serverData.fen_after_human);
        addMoveToLog(serverData.human_san, 'human', serverData.fen_after_human);
        removeHighlights();

        // This universally handles live player check highlighting!
        if (serverData.in_check_after_human) highlightSquare(serverData.king_square_after_human, 'check-square');

        if (serverData.status_after_human === 'over') {
            gameOver = true; stopClock();
            saveGameToLocal(serverData.game_summary);
            showResultModal(serverData.is_draw ? 'Draw' : 'Game Over', serverData.msg || 'Checkmate');
            return;
        }

        // Toggle turn color and start the clock!
        currentTurnColor = (currentTurnColor === 'white') ? 'black' : 'white';

        if (gameMode === 'ai') {
            setThinking(true);
            if (clockEnabled) startClock(currentTurnColor);
            fetchAIMove();
        } else {
            isPlayerTurn = true;
            if (clockEnabled) startClock(currentTurnColor);
        }
    }

    animatePieceMove(from, to, ANIM, () => { animDone = true; tryFinalize(); });
    fetch('/human_move', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({move:uci}) })
    .then(r => r.json())
    .then(data => { serverData = data; tryFinalize(); })
    .catch(err => { console.error('Move error:', err); isPlayerTurn = true; });
}

// ── Fetch AI move ─────────────────────────────────────────────────────
function fetchAIMove() {
    fetch('/ai_move', { method:'POST', headers:{'Content-Type':'application/json'} })
    .then(r => r.json())
    .then(data => {
        setThinking(false);
        if (!data.ai_san) {
            board.position(data.fen, false);
            if (data.status === 'over') {
                gameOver=true; stopClock();
                saveGameToLocal(data.game_summary);
                showResultModal(data.is_draw?'Draw':'Game Over', data.msg||'Game ended');
                }
            else { isPlayerTurn=true; if (clockEnabled) startClock(playerColor); }
            return;
        }
        var from = data.ai_uci ? data.ai_uci.slice(0,2) : null;
        var to   = data.ai_uci ? data.ai_uci.slice(2,4) : null;

        function finalize() {
            board.position(data.fen, false);
            fenHistory.push(data.fen);
            removeHighlights();
            addMoveToLog(data.ai_san, 'ai', data.fen);

            // This universally handles live AI check highlighting!
            if (data.in_check) highlightSquare(data.king_square, 'check-square');

            if (data.status === 'over') {
                gameOver=true; stopClock();
                saveGameToLocal(data.game_summary);
                showResultModal(data.is_draw?'Draw':'Game Over', data.msg||'Checkmate');
            } else {
                // Toggle back to human and start clock
                currentTurnColor = (currentTurnColor === 'white') ? 'black' : 'white';
                isPlayerTurn = true;
                if (clockEnabled) startClock(currentTurnColor);
            }
        }
        if (from && to) animatePieceMove(from, to, 380, finalize); else finalize();
    })
    .catch(err => { console.error('ai_move error:', err); setThinking(false); isPlayerTurn=true; if(clockEnabled) startClock(playerColor); });
}

// ── Move log ──────────────────────────────────────────────────────────
function addMoveToLog(san, side, fen) {
    const log = document.getElementById('move-log');
    if (!log || !san) return;
    const isWhiteMove  = (moveCount % 2 === 1);
    const fullMoveNum  = Math.ceil(moveCount / 2);
    const historyIndex = fenHistory.length - 1;

    if (isWhiteMove) {
        const row = document.createElement('div');
        row.className = 'move-row';
        row.dataset.moveNum = fullMoveNum;

        const numSpan = document.createElement('span');
        numSpan.className = 'move-num-label';
        numSpan.textContent = fullMoveNum + '.';

        const whiteSpan = document.createElement('span');
        whiteSpan.className      = 'move-half move-white' + (side === 'ai' ? ' move-ai' : '');
        whiteSpan.id             = 'move-white-' + fullMoveNum;
        whiteSpan.textContent    = san;
        whiteSpan.dataset.fenIdx = historyIndex;
        whiteSpan.title          = 'Click to view position';
        whiteSpan.addEventListener('click', () => jumpToFen(parseInt(whiteSpan.dataset.fenIdx)));

        const blackSpan = document.createElement('span');
        blackSpan.className   = 'move-half move-black-placeholder';
        blackSpan.id          = 'move-black-' + fullMoveNum;
        blackSpan.textContent = '';

        row.appendChild(numSpan); row.appendChild(whiteSpan); row.appendChild(blackSpan);
        log.appendChild(row);
    } else {
        const blackEl = document.getElementById('move-black-' + fullMoveNum);
        if (blackEl) {
            blackEl.className      = 'move-half move-black' + (side === 'ai' ? ' move-ai' : '');
            blackEl.textContent    = san;
            blackEl.dataset.fenIdx = historyIndex;
            blackEl.title          = 'Click to view position';
            blackEl.addEventListener('click', () => jumpToFen(parseInt(blackEl.dataset.fenIdx)));
        }
    }
    log.scrollTop = log.scrollHeight;
    moveCount++;
}

// ── Move replay (#14) ─────────────────────────────────────────────────
function jumpToFen(idx) {
    if (!fenHistory.length || idx < 0 || idx >= fenHistory.length) return;
    replayMode = true; replayIndex = idx;
    board.position(fenHistory[idx], false);

    // Clear check highlights when viewing past positions
    removeHighlights();

    document.querySelectorAll('.move-half[data-fen-idx]').forEach(el => {
        el.classList.toggle('move-replay-active', parseInt(el.dataset.fenIdx) === idx);
    });
    showReplayBanner();
}

function exitReplayMode() {
    if (!replayMode) return;
    replayMode = false; replayIndex = -1;
    const liveFen = fenHistory[fenHistory.length - 1];
    if (liveFen) board.position(liveFen, false);
    document.querySelectorAll('.move-half[data-fen-idx]').forEach(el => el.classList.remove('move-replay-active'));
    hideReplayBanner();
}

function showReplayBanner() {
    let b = document.getElementById('replay-banner');
    if (!b) {
        b = document.createElement('div');
        b.id = 'replay-banner'; b.className = 'replay-banner';
        b.innerHTML = '⏪ Replay mode — <span>click board to return to live</span>';
        document.querySelector('.board-container').prepend(b);
    }
    b.style.display = 'block';
}
function hideReplayBanner() { const b = document.getElementById('replay-banner'); if (b) b.style.display = 'none'; }

// ── Thinking indicator ────────────────────────────────────────────────
function setThinking(on) {
    [document.getElementById('white-name'), document.getElementById('black-name')].forEach(el => {
        if (!el || el.dataset.isAi !== 'true') return;
        el.innerHTML = on ? el.dataset.name + ' <span class="thinking-tag">Thinking\u2026</span>' : el.dataset.name;
    });
}

// ── Highlights ────────────────────────────────────────────────────────
function deselectPiece() { selectedSquare=null; validMovesForPiece=[]; removeHighlights(); }
// This guarantees .check-square is removed universally!
function removeHighlights() { $('.square-55d63').removeClass('selected-square legal-highlight check-square'); }
function highlightSquare(sq, cls) { if (sq) $('#board .square-' + sq).addClass(cls); }

// ── Resign ────────────────────────────────────────────────────────────
function showResignModal()  { if (!gameOver) document.getElementById('resign-modal').style.display = 'flex'; }
function closeResignModal() { document.getElementById('resign-modal').style.display = 'none'; }

function executeResign() {
    closeResignModal();
    fetch('/resign', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({mode:gameMode}) })
    .then(r => r.json())
    .then(data => {
        gameOver=true; isPlayerTurn=false; stopClock(); setThinking(false);
        saveGameToLocal(data.game_summary);
        showResultModal('Resignation', data.msg);
    });
}

// ── Home button ───────────────────────────────────────────────────────
function handleHomeClick() {
    // Only ask for resign if the game is active AND at least one full turn has happened
    // fenHistory.length <= 2 means at most 1 half-move has been played
    if (!gameOver && fenHistory.length > 2) {
        showResignModal();
    } else {
        // If the game is already over OR barely started, allow instant navigation
        window.location.href = '/';
    }
}

// ── Result modal ──────────────────────────────────────────────────────
function showResultModal(title, msg) {
    document.getElementById('result-title').innerText = title;
    document.getElementById('result-body').innerText  = msg;
    document.getElementById('result-modal').style.display = 'flex';
}
function goToHome() { window.location.href = '/'; }

// ── PGN ───────────────────────────────────────────────────────────────
function downloadPGN() { window.location.href = '/download_pgn'; }

// FIX #2: Use URL as fallback when localStorage is full or unavailable
function openAnalysis() {
    fetch('/get_pgn').then(r => r.json()).then(data => {
        if (data.pgn) {
            let stored = false;
            try {
                localStorage.setItem('ghost_ai_pgn_transfer', data.pgn);
                stored = true;
            } catch(e) {
                console.warn('localStorage full, falling back to URL transfer:', e);
            }
            // If localStorage succeeded navigate normally; otherwise pass PGN via URL
            window.location.href = stored
                ? '/analysis'
                : '/analysis?pgn=' + encodeURIComponent(data.pgn);
        } else {
            window.location.href = '/analysis';
        }
    }).catch(() => { window.location.href = '/analysis'; });
}

// ── Board click wiring ────────────────────────────────────────────────
$(document).on('click', '.square-55d63', function () { onSquareClick($(this).attr('data-square')); });

// ── Pawn Promotion UI ─────────────────────────────────────────────────
function showInlinePromotion(color, targetSquare) {
    const backdrop  = document.getElementById('promo-backdrop');
    const container = document.getElementById('inline-promo');
    container.innerHTML = '';

    const pieces = ['q', 'n', 'r', 'b'];
    pieces.forEach(p => {
        const div = document.createElement('div');
        div.className = 'inline-promo-piece';
        div.onclick = (e) => { e.stopPropagation(); submitPromotion(p); };
        const img = document.createElement('img');
        img.src = `/Chess_AI/images/${color}${p.toUpperCase()}.png`;
        div.appendChild(img);
        container.appendChild(div);
    });

    const sqEl = document.querySelector(`.square-${targetSquare}`);
    if (sqEl) {
        const rect = sqEl.getBoundingClientRect();

        container.style.visibility = 'hidden';
        container.style.animation  = 'none';
        container.style.display    = 'flex';
        container.style.flexDirection = 'row';

        const contRect  = container.getBoundingClientRect();
        const safeWidth  = contRect.width  > 150 ? contRect.width  : 220;
        const safeHeight = contRect.height > 40  ? contRect.height : 60;

        container.style.visibility = 'visible';
        container.style.animation  = '';

        const vWidth  = document.documentElement.clientWidth  || window.innerWidth;
        const vHeight = document.documentElement.clientHeight || window.innerHeight;

        let top = rect.top - safeHeight - 8;
        let left;

        if (vWidth <= 900) {
            const isPhysicallyRight = (rect.left + rect.width / 2) > (vWidth / 2);
            left = isPhysicallyRight ? (rect.right - safeWidth) : rect.left;
        } else {
            const file = targetSquare[0];
            const isRightSide = ['e', 'f', 'g', 'h'].includes(file);
            left = isRightSide ? (rect.right - safeWidth) : rect.left;
        }

        if (left + safeWidth > vWidth - 10) left = vWidth - safeWidth - 10;
        if (left < 10) left = 10;
        if (top < 10)  top  = rect.bottom + 8;
        if (top + safeHeight > vHeight - 10) top = vHeight - safeHeight - 10;

        container.style.top  = top  + 'px';
        container.style.left = left + 'px';
    } else {
        container.style.display       = 'flex';
        container.style.flexDirection = 'row';
        container.style.top           = '50%';
        container.style.left          = '50%';
        container.style.transform     = 'translate(-50%, -50%)';
    }
    backdrop.style.display = 'block';
}

function cancelPromotion() {
    document.getElementById('promo-backdrop').style.display   = 'none';
    document.getElementById('inline-promo').style.display     = 'none';
    document.getElementById('inline-promo').style.transform   = '';
    pendingPromo = null;
    isPlayerTurn = true;
}

function submitPromotion(promoPiece) {
    document.getElementById('promo-backdrop').style.display = 'none';
    document.getElementById('inline-promo').style.display   = 'none';
    document.getElementById('inline-promo').style.transform = '';

    if (!pendingPromo) return;
    const uci = pendingPromo.uciBase + promoPiece;  // e.g. 'e7e8q'
    pendingPromo = null;
    executeMove(uci);
}

// ── Inject Promotion UI HTML + CSS (self-contained) ───────────────────
(function injectPromoUI() {
    // CSS
    const style = document.createElement('style');
    style.textContent = `
        .promo-backdrop {
            position: fixed; inset: 0; z-index: 9998; background: transparent; display: none;
        }
        .inline-promo {
            position: fixed; display: none; flex-direction: row;
            background: var(--panel2, #1a1a1a);
            border: 2px solid var(--gold, #c9a84c); border-radius: 8px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.8);
            z-index: 9999; padding: 5px; gap: 5px;
            animation: promo-pop-in 0.15s cubic-bezier(0.18, 0.89, 0.32, 1.28);
        }
        @keyframes promo-pop-in {
            from { transform: scale(0.8); opacity: 0; }
            to   { transform: scale(1);   opacity: 1; }
        }
        .inline-promo-piece {
            width: 48px; height: 48px; background: #111;
            border: 1px solid var(--border, #333);
            border-radius: 6px; cursor: pointer;
            transition: background 0.15s, border-color 0.15s, transform 0.1s;
            display: flex; align-items: center; justify-content: center;
        }
        .inline-promo-piece:hover {
            background: #242010; border-color: var(--gold, #c9a84c); transform: scale(1.08);
        }
        .inline-promo-piece img { width: 40px; height: 40px; pointer-events: none; }
    `;
    document.head.appendChild(style);

    // HTML
    const backdrop = document.createElement('div');
    backdrop.id        = 'promo-backdrop';
    backdrop.className = 'promo-backdrop';
    backdrop.onclick   = () => cancelPromotion();

    const container = document.createElement('div');
    container.id        = 'inline-promo';
    container.className = 'inline-promo';
    container.onclick   = (e) => e.stopPropagation();

    backdrop.appendChild(container);
    document.body.appendChild(backdrop);
})();
function saveGameToLocal(summary) {
    if (!summary) return;

    let history = [];
    try {
        history = JSON.parse(localStorage.getItem('ghost_history')) || [];
    } catch(e) {
        history = [];
    }

    history.unshift(summary);

    // Enforce the 50-game cap advertised in the History UI
    if (history.length > 50) {
        history = history.slice(0, 50);
    }

    try {
        localStorage.setItem('ghost_history', JSON.stringify(history));
    } catch(e) {
        // Storage quota exceeded — trim to half and retry once
        console.warn('localStorage quota exceeded when saving history, trimming…', e);
        try {
            history = history.slice(0, Math.floor(history.length / 2));
            localStorage.setItem('ghost_history', JSON.stringify(history));
        } catch(e2) {
            console.error('Unable to save game history even after trimming:', e2);
        }
    }
}