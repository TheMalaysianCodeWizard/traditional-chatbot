/* TradBot - browser (JavaScript) version of the traditional chatbot engine.
 *
 * Hugging Face Static Spaces serve only HTML/JS, so this file is a
 * line-by-line port of the Python engine in ../chatbot/. It uses the same
 * knowledge base (intents.js, generated from chatbot/intents.json), the same
 * input validation, the same regex + TF-IDF cascade (word 1-2 grams and
 * char_wb 2-4 grams, smoothed idf, sublinear tf, L2 norm - scikit-learn's
 * formulas), and the same thresholds. tests/parity_test.js checks that both
 * versions pick the same intent with the same confidence.
 *
 * Works in the browser (window.TradBot) and in Node (module.exports).
 */
(function (root) {
  "use strict";

  const MAX_LEN = 500;
  const CONFIDENT = 0.45; // at or above: answer directly
  const CLARIFY = 0.25;   // between CLARIFY and CONFIDENT: ask "did you mean ...?"
  const TIMEZONE = "America/Chicago";

  // ------------------------------------------------------------- NLU: rules
  class RegexProvider {
    constructor(intents) {
      this.name = "regex";
      this.rules = [];
      for (const it of intents) for (const p of it.patterns || []) this.rules.push([it.tag, new RegExp(p, "i")]);
    }
    predict(text) {
      for (const [tag, rx] of this.rules) if (rx.test(text)) return { tag, confidence: 1.0, source: this.name };
      return { tag: null, confidence: 0, source: this.name };
    }
  }

  // ------------------------------------------------------------ NLU: TF-IDF
  const wordAnalyzer = (text) => {
    const toks = text.toLowerCase().match(/[\p{L}\p{N}_]{2,}/gu) || [];
    const grams = toks.slice();
    for (let i = 0; i + 1 < toks.length; i++) grams.push(toks[i] + " " + toks[i + 1]);
    return grams;
  };
  const charWbAnalyzer = (text) => {
    const grams = [];
    const words = text.toLowerCase().replace(/\s\s+/g, " ").split(/\s+/).filter(Boolean);
    for (const raw of words) {
      const w = [..." " + raw + " "];
      for (let n = 2; n <= 4; n++) {
        let offset = 0;
        grams.push(w.slice(offset, offset + n).join(""));
        while (offset + n < w.length) { offset += 1; grams.push(w.slice(offset, offset + n).join("")); }
        if (offset === 0) break;
      }
    }
    return grams;
  };

  class Tfidf {
    constructor(docs, analyzer) {
      this.analyzer = analyzer;
      const df = new Map();
      const counted = docs.map((d) => this.count(d));
      for (const c of counted) for (const k of c.keys()) df.set(k, (df.get(k) || 0) + 1);
      const n = docs.length;
      this.idf = new Map();
      for (const [k, v] of df) this.idf.set(k, Math.log((1 + n) / (1 + v)) + 1);
      this.vectors = counted.map((c) => this.weigh(c));
    }
    count(text) {
      const m = new Map();
      for (const g of this.analyzer(text)) m.set(g, (m.get(g) || 0) + 1);
      return m;
    }
    weigh(counts) {
      const v = new Map();
      let norm = 0;
      for (const [k, c] of counts) {
        if (!this.idf.has(k)) continue;
        const w = (1 + Math.log(c)) * this.idf.get(k);
        v.set(k, w); norm += w * w;
      }
      norm = Math.sqrt(norm);
      if (norm > 0) for (const [k, w] of v) v.set(k, w / norm);
      return v;
    }
    similarities(text) {
      const q = this.weigh(this.count(text));
      return this.vectors.map((d) => {
        let s = 0;
        for (const [k, w] of q) if (d.has(k)) s += w * d.get(k);
        return s;
      });
    }
  }

  class TfidfProvider {
    constructor(intents) {
      this.name = "tfidf";
      this.examples = []; this.labels = [];
      for (const it of intents) for (const ex of it.examples || []) { this.examples.push(ex.toLowerCase()); this.labels.push(it.tag); }
      this.word = new Tfidf(this.examples, wordAnalyzer);
      this.char = new Tfidf(this.examples, charWbAnalyzer);
    }
    predict(text) {
      const t = text.toLowerCase();
      const sw = this.word.similarities(t), sc = this.char.similarities(t);
      const best = new Map();
      this.labels.forEach((label, i) => {
        const s = 0.5 * sw[i] + 0.5 * sc[i];
        if (!best.has(label) || s > best.get(label)) best.set(label, s);
      });
      const ranked = [...best.entries()].sort((a, b) => b[1] - a[1]);
      if (!ranked.length) return { tag: null, confidence: 0, source: this.name };
      return { tag: ranked[0][0], confidence: ranked[0][1], source: this.name, alternatives: ranked.slice(1, 3) };
    }
  }

  class CascadeProvider {
    constructor(providers) { this.name = "cascade"; this.providers = providers; }
    predict(text) {
      let best = { tag: null, confidence: 0, source: this.name };
      for (const p of this.providers) {
        const r = p.predict(text);
        if (r.tag && r.confidence >= 1.0) return r; // an exact rule hit wins immediately
        if (r.confidence > best.confidence) best = r;
      }
      return best;
    }
  }

  // ------------------------------------------------------------------ skills
  const CAPABILITIES = [
    ["Greetings and small talk", "hi, how are you, thanks, bye"],
    ["Tell you about myself", "who are you, how do you work"],
    ["Remember your name", "my name is Sam  /  what is my name"],
    ["Tell the time and date", "what time is it, what is the date"],
    ["Do arithmetic", "calculate (3 + 4) * 2, what is 2^10"],
    ["Tell a joke", "tell me a joke"],
    ["Echo or reverse text", "echo hello  /  reverse hello"],
    ["Explain chatbot concepts", "chatbot lifecycle, traditional vs LLM"],
    ["Show this list", "help"],
  ];

  const capabilities = () =>
    ["Here's what I can do:", ...CAPABILITIES.map(([s, ex], i) => `${i + 1}. ${s} - try: "${ex}"`)].join("\n");

  function setName(text, session) {
    const m = text.match(/(?:my name is|call me|i am called|i'm called)\s+([a-z][a-z' -]{0,30})/i);
    if (!m) return "I didn't catch your name. Try: \"my name is Sam\".";
    const first = m[1].trim().split(/\s+/)[0];
    const name = first.charAt(0).toUpperCase() + first.slice(1).toLowerCase();
    session.name = name;
    return `Nice to meet you, ${name}! I'll remember that for this conversation.`;
  }
  const getName = (_t, s) => s.name ? `Your name is ${s.name}.` : "I don't know your name yet. Tell me with \"my name is ...\".";
  const time = () => `It's ${new Intl.DateTimeFormat("en-US", { timeZone: TIMEZONE, hour: "numeric", minute: "2-digit" }).format(new Date())} (US Central time).`;
  const date = () => `Today is ${new Intl.DateTimeFormat("en-US", { timeZone: TIMEZONE, weekday: "long", month: "long", day: "2-digit", year: "numeric" }).format(new Date())}.`;

  // Safe arithmetic: a tiny recursive-descent parser (never eval).
  // Grammar mirrors Python: expr := term (+|- term)* ; term := factor (*|/|% factor)* ;
  // factor := (+|-) factor | power ; power := atom [** factor] ; atom := number | ( expr )
  function evaluate(src) {
    const toks = src.match(/\*\*|\d+\.?\d*|\.\d+|[-+*/%()]|\S/g) || [];
    let i = 0;
    const peek = () => toks[i], next = () => toks[i++];
    const fail = (msg) => { throw new SyntaxError(msg || "malformed"); };
    function atom() {
      const t = next();
      if (t === undefined) fail();
      if (t === "(") { const v = expr(); if (next() !== ")") fail(); return v; }
      if (/^(\d+\.?\d*|\.\d+)$/.test(t)) return parseFloat(t);
      fail();
    }
    function power() {
      const base = atom();
      if (peek() === "**") {
        next(); const e = factor();
        if (Math.abs(e) > 100) fail("exponent too large");
        if (base === 0 && e < 0) throw new RangeError("zero");
        return Math.pow(base, e);
      }
      return base;
    }
    function factor() {
      if (peek() === "-") { next(); return -factor(); }
      if (peek() === "+") { next(); return +factor(); }
      return power();
    }
    function term() {
      let v = factor();
      while (["*", "/", "%"].includes(peek())) {
        const op = next(), r = factor();
        if (op === "*") v *= r;
        else if (r === 0) throw new RangeError("zero");
        else if (op === "/") v /= r;
        else v = ((v % r) + r) % r; // Python-style floored modulo
      }
      return v;
    }
    function expr() {
      let v = term();
      while (peek() === "+" || peek() === "-") v = next() === "+" ? v + term() : v - term();
      return v;
    }
    const v = expr();
    if (i !== toks.length) fail();
    return v;
  }

  const WORDS = { plus: "+", minus: "-", times: "*", "multiplied by": "*", x: "*", "divided by": "/", over: "/", mod: "%", "to the power of": "**" };
  function math(text) {
    let expr = text.trim().replace(/^(calculate|compute|calc|math|what is|what's)\b\s*/i, "");
    expr = expr.replace(/[?=. ]+$/, "").toLowerCase();
    for (const w of Object.keys(WORDS).sort((a, b) => b.length - a.length)) expr = expr.replace(new RegExp(`\\b${w}\\b`, "g"), WORDS[w]);
    expr = expr.replace(/\^/g, "**");
    if (!expr || !/^[\d\s.+\-*/()%]+$/.test(expr))
      return "I can only do arithmetic with numbers and + - * / % ^ ( ). Example: calculate (3 + 4) * 2";
    let value;
    try { value = evaluate(expr); }
    catch (e) {
      if (e instanceof RangeError) return "Dividing by zero isn't defined, so I can't calculate that.";
      return "That expression looks malformed. Check the brackets and operators, e.g. calculate (3 + 4) * 2";
    }
    const shown = Number.isInteger(value) ? String(value) : String(parseFloat(value.toFixed(6)));
    return `${expr.replace(/\*\*/g, "^")} = ${shown}`;
  }
  const payload = (text, cmd) => text.trim().replace(new RegExp(`^${cmd}\\s*:?\\s*`, "i"), "");
  const echo = (t) => { const p = payload(t, "echo"); return p ? `You said: ${p}` : "Give me something to echo, e.g. \"echo hello\"."; };
  const reverse = (t) => { const p = payload(t, "reverse"); return p ? [...p].reverse().join("") : "Give me something to reverse, e.g. \"reverse hello\"."; };

  const HANDLERS = { capabilities, set_name: setName, get_name: getName, time, date, math, echo, reverse };

  // ------------------------------------------------------------------ engine
  const FRIENDLY = {
    greeting: "say hello", how_are_you: "ask how I'm doing", capabilities: "see what I can do",
    bot_identity: "know who I am", how_do_you_work: "know how I work", set_name: "tell me your name",
    get_name: "ask what your name is", time: "know the time", date: "know today's date",
    math: "do a calculation", joke: "hear a joke", echo: "have me echo something",
    reverse: "reverse some text", chatbot_lifecycle: "learn the chatbot development lifecycle",
    traditional_vs_llm: "compare traditional bots and LLMs", thanks: "say thanks", goodbye: "say goodbye",
  };
  const MALFORMED = {
    empty: "It looks like your message was empty. Type something, or 'help' to see what I can do.",
    too_long: `That message is too long for me (limit ${MAX_LEN} characters). Could you shorten it?`,
    no_words: "I can only read letters and numbers, and that message didn't have any. Try typing a question.",
    gibberish: "Hmm, that doesn't look like words I know. Could you rephrase? Type 'help' for examples.",
  };
  const YES = /^(y|ya|yes|yeah|yep|sure|ok|okay|correct|right)\b/i;
  const NO = /^(n|no|nope|nah|not really|wrong)\b/i;

  function clean(raw) {
    if (raw === null || raw === undefined) return [null, "empty"];
    let text = String(raw).normalize("NFKC");
    text = text.replace(/\p{C}/gu, "");                 // control/format characters
    text = text.replace(/<[^>]+>/g, " ");               // strip HTML/markup tags
    text = text.replace(/\s+/g, " ").trim();
    if (!text) return [null, "empty"];
    if ([...text].length > MAX_LEN) return [null, "too_long"];
    if (!/[A-Za-z0-9]/.test(text)) return [null, "no_words"];
    text = text.replace(/([^\d])\1{2,}/gu, "$1");        // "heyyyy!!!" -> "hey!"
    const letters = text.match(/[A-Za-z]+/g) || [];
    if (letters.length && !/\d/.test(text)) {
      const long = letters.filter((w) => w.length >= 5);
      const noVowel = long.filter((w) => !/[aeiouy]/i.test(w));
      if (long.length && noVowel.length / long.length > 0.5) return [null, "gibberish"];
    }
    return [text, null];
  }

  class ChatBot {
    constructor(data, nlu) {
      this.intents = Object.fromEntries(data.intents.map((i) => [i.tag, i]));
      this.nlu = nlu || new CascadeProvider([new RegexProvider(data.intents), new TfidfProvider(data.intents)]);
    }
    respond(raw, session) {
      session.misses = session.misses || 0;
      session.log = session.log || [];
      const [text, problem] = clean(raw);
      if (problem) return this.log(session, raw, { text: MALFORMED[problem], intent: null, confidence: 0, source: `validator:${problem}` });

      const pending = session.pending; delete session.pending;
      if (pending) {
        if (YES.test(text)) return this.log(session, raw, this.answer(pending.intent, pending.text, session, 1.0, "clarified"));
        if (NO.test(text)) return this.log(session, raw, { text: "OK, my mistake. Could you say it another way? Type 'help' to see examples.", intent: null, confidence: 0, source: "clarified" });
      }

      const r = this.nlu.predict(text);
      if (r.tag && r.confidence >= CONFIDENT) {
        session.misses = 0;
        return this.log(session, raw, this.answer(r.tag, text, session, r.confidence, r.source));
      }
      if (r.tag && r.confidence >= CLARIFY) {
        session.pending = { intent: r.tag, text };
        return this.log(session, raw, { text: `I'm not completely sure. Did you want to ${FRIENDLY[r.tag] || r.tag}? (yes/no)`, intent: r.tag, confidence: r.confidence, source: r.source });
      }
      session.misses += 1;
      let msg;
      if (session.misses >= 2) { session.misses = 0; msg = "I'm still not following, sorry. " + capabilities(); }
      else msg = pick([
        "Sorry, I didn't understand that. Could you rephrase? Type 'help' to see what I can do.",
        "I'm not sure what you mean. I'm a simple rule-based bot; type 'help' for things I understand.",
      ]);
      return this.log(session, raw, { text: msg, intent: null, confidence: r.confidence, source: "fallback" });
    }
    answer(tag, text, session, confidence, source) {
      const intent = this.intents[tag] || {};
      const handler = HANDLERS[intent.handler];
      let body;
      if (handler) body = handler(text, session);
      else if (intent.responses && intent.responses.length) body = pick(intent.responses);
      else body = "I know what you mean, but I don't have an answer for that yet.";
      if (tag === "greeting" && session.name)
        body = body.replace("Hello!", `Hello, ${session.name}!`).replace("Hi there!", `Hi ${session.name}!`);
      return { text: body, intent: tag, confidence, source };
    }
    log(session, raw, reply) {
      session.log.push({ user: raw, intent: reply.intent, confidence: Math.round(reply.confidence * 1000) / 1000, source: reply.source });
      return reply;
    }
  }
  const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];

  const api = { ChatBot, clean, evaluate, RegexProvider, TfidfProvider, CascadeProvider };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TradBot = api;
})(typeof window !== "undefined" ? window : globalThis);
