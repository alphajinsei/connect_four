/**
 * game.js — Connect Three のゲームロジック + DQN推論（ブラウザ版）
 *
 * Python版からの移植元:
 *   env/connect3_env.py     … 盤面・重力・勝利判定・状態表現
 *   agents/dqn_agent.py     … get_action（無効手マスキング + argmax）
 *
 * 重要: Python版と同じ手を打つ必要がある。特に以下は忠実に移植すること。
 *   - get_state のチャンネル構成（ch0=自分, ch1=相手, ch2=手番）
 *   - 無効手を -Infinity でマスクしてから argmax する手順
 */

const ROWS = 5;
const COLS = 5;
const WIN_LENGTH = 3;

const EMPTY = 0;
const PLAYER1 = 1;   // AI（DQN）— 先手固定で学習しているため常にこちら
const PLAYER2 = -1;  // 人間

class Connect3Env {
  constructor() {
    this.reset();
  }

  reset() {
    this.board = Array.from({ length: ROWS }, () => new Array(COLS).fill(EMPTY));
    this.currentPlayer = PLAYER1;
    this.done = false;
    this.winner = null;
  }

  getValidActions() {
    const valid = [];
    for (let c = 0; c < COLS; c++) {
      if (this.board[0][c] === EMPTY) valid.push(c);
    }
    return valid;
  }

  /** 重力: 列 col で駒が落ちる行を返す。満杯なら null。 */
  applyGravity(col) {
    for (let r = ROWS - 1; r >= 0; r--) {
      if (this.board[r][col] === EMPTY) return r;
    }
    return null;
  }

  step(col) {
    if (this.done) return;
    const row = this.applyGravity(col);
    if (row === null) return;

    this.board[row][col] = this.currentPlayer;

    if (this.checkWin(this.currentPlayer)) {
      this.done = true;
      this.winner = this.currentPlayer;
      return;
    }
    if (this.getValidActions().length === 0) {
      this.done = true;
      this.winner = 0; // 引き分け
      return;
    }
    this.currentPlayer = this.currentPlayer === PLAYER1 ? PLAYER2 : PLAYER1;
  }

  checkWin(player) {
    const b = this.board;
    const wl = WIN_LENGTH;
    // 横
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c <= COLS - wl; c++) {
        let ok = true;
        for (let i = 0; i < wl; i++) if (b[r][c + i] !== player) { ok = false; break; }
        if (ok) return true;
      }
    }
    // 縦
    for (let r = 0; r <= ROWS - wl; r++) {
      for (let c = 0; c < COLS; c++) {
        let ok = true;
        for (let i = 0; i < wl; i++) if (b[r + i][c] !== player) { ok = false; break; }
        if (ok) return true;
      }
    }
    // 斜め（右下）
    for (let r = 0; r <= ROWS - wl; r++) {
      for (let c = 0; c <= COLS - wl; c++) {
        let ok = true;
        for (let i = 0; i < wl; i++) if (b[r + i][c + i] !== player) { ok = false; break; }
        if (ok) return true;
      }
    }
    // 斜め（左下）
    for (let r = 0; r <= ROWS - wl; r++) {
      for (let c = wl - 1; c < COLS; c++) {
        let ok = true;
        for (let i = 0; i < wl; i++) if (b[r + i][c - i] !== player) { ok = false; break; }
        if (ok) return true;
      }
    }
    return false;
  }

  /**
   * NN入力用の状態を (3, ROWS, COLS) のフラット配列で返す。
   * Python版 get_state と同じ構成:
   *   ch0: perspective のコマ
   *   ch1: 相手のコマ
   *   ch2: perspective の手番なら 1.0
   */
  getState(perspective) {
    const opponent = perspective === PLAYER1 ? PLAYER2 : PLAYER1;
    const data = new Float32Array(3 * ROWS * COLS);
    const turnFlag = this.currentPlayer === perspective ? 1.0 : 0.0;
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const v = this.board[r][c];
        const idx = r * COLS + c;
        if (v === perspective) data[idx] = 1.0;
        else if (v === opponent) data[ROWS * COLS + idx] = 1.0;
        data[2 * ROWS * COLS + idx] = turnFlag;
      }
    }
    return data;
  }
}

/** ONNXモデルを読み込んで推論するエージェント（Python版 DQNAgent.get_action 相当、ε=0） */
class DQNAgent {
  constructor(session) {
    this.session = session;
  }

  static async create(modelUrl) {
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
    });
    return new DQNAgent(session);
  }

  async getAction(state, validActions) {
    const tensor = new ort.Tensor('float32', state, [1, 3, ROWS, COLS]);
    const out = await this.session.run({ board: tensor });
    const q = out.q_values.data;

    // 無効手を -Infinity でマスクしてから argmax（Python版と同じ手順）
    let bestCol = validActions[0];
    let bestQ = -Infinity;
    for (const c of validActions) {
      if (q[c] > bestQ) {
        bestQ = q[c];
        bestCol = c;
      }
    }
    return bestCol;
  }
}

/**
 * Lv.0「学習前」用のエージェント。合法手から一様ランダムに選ぶ。
 * DQNAgent と同じインターフェースを持たせ、UI側で分岐しないようにする。
 */
class RandomAgent {
  async getAction(state, validActions) {
    return validActions[Math.floor(Math.random() * validActions.length)];
  }
}
