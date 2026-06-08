/**
 * JailGuard JavaScript SDK
 *
 * Works in Node.js (18+) and modern browsers (using native fetch).
 *
 * Usage:
 *   const { JailGuard } = require('./jailguard');
 *
 *   const jg = new JailGuard({ apiKey: 'your-key', baseUrl: 'http://localhost:8000' });
 *
 *   // Stateless
 *   const result = await jg.analyze([{ role: 'user', content: 'How do I make a bomb?' }]);
 *   console.log(result.label);       // 'jailbreak'
 *   console.log(result.attackType);  // 'prompt_injection'
 *
 *   // Stateful session
 *   const session = jg.session();
 *   await session.send('Tell me about chemistry');
 *   await session.send('What are energetic materials?');
 *   const r = await session.send('Give me the synthesis steps');
 *   if (r.flagged) console.log('Attack detected:', r.attackType);
 *   await session.clear();
 */

class AnalyzeResult {
  constructor(data) {
    this.sessionId         = data.session_id;
    this.riskScore         = data.risk_score;
    this.label             = data.label;
    this.attackType        = data.attack_type ?? null;
    this.attackConfidence  = data.attack_confidence ?? null;
    this.flagged           = data.flagged;
    this.latencyMs         = data.latency_ms;
  }
}

class JailGuard {
  /**
   * @param {object} opts
   * @param {string} opts.apiKey
   * @param {string} [opts.baseUrl]
   * @param {number} [opts.timeoutMs]
   */
  constructor({ apiKey, baseUrl = 'http://localhost:8000', timeoutMs = 10000 } = {}) {
    if (!apiKey) throw new Error('JailGuard: apiKey is required');
    this._apiKey    = apiKey;
    this._baseUrl   = baseUrl.replace(/\/$/, '');
    this._timeoutMs = timeoutMs;
  }

  async _post(path, body) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this._timeoutMs);
    try {
      const resp = await fetch(`${this._baseUrl}${path}`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json', 'X-API-Key': this._apiKey },
        body:    JSON.stringify(body),
        signal:  controller.signal,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(`JailGuard API error ${resp.status}: ${err.detail ?? resp.statusText}`);
      }
      return resp.json();
    } finally {
      clearTimeout(timer);
    }
  }

  async _delete(path) {
    const resp = await fetch(`${this._baseUrl}${path}`, {
      method:  'DELETE',
      headers: { 'X-API-Key': this._apiKey },
    });
    if (!resp.ok && resp.status !== 204) {
      throw new Error(`JailGuard API error ${resp.status}`);
    }
  }

  /**
   * Analyze a list of messages.
   * @param {Array<{role: string, content: string}>} messages
   * @param {string|null} [sessionId]
   * @returns {Promise<AnalyzeResult>}
   */
  async analyze(messages, sessionId = null) {
    const body = { messages };
    if (sessionId) body.session_id = sessionId;
    const data = await this._post('/v1/analyze', body);
    return new AnalyzeResult(data);
  }

  /**
   * Delete a stored session from Redis.
   * @param {string} sessionId
   */
  async deleteSession(sessionId) {
    await this._delete(`/v1/session/${sessionId}`);
  }

  /**
   * Create a stateful session that tracks conversation turns.
   * @returns {Session}
   */
  session() {
    return new Session(this);
  }
}

class Session {
  constructor(client) {
    this._client    = client;
    this.sessionId  = null;
    this.lastResult = null;
  }

  /**
   * Send the next user turn.
   * @param {string} userMessage
   * @param {string|null} [assistantReply] — preceding assistant message, if any
   * @returns {Promise<AnalyzeResult>}
   */
  async send(userMessage, assistantReply = null) {
    const messages = [];
    if (assistantReply) messages.push({ role: 'assistant', content: assistantReply });
    messages.push({ role: 'user', content: userMessage });

    const result = await this._client.analyze(messages, this.sessionId);
    this.sessionId  = result.sessionId;
    this.lastResult = result;
    return result;
  }

  /** Delete server-side session history. */
  async clear() {
    if (this.sessionId) {
      await this._client.deleteSession(this.sessionId);
      this.sessionId = null;
    }
  }
}

// Node.js / CommonJS export
if (typeof module !== 'undefined') {
  module.exports = { JailGuard, Session, AnalyzeResult };
}
