/* Checks that the JavaScript port (hf_space/bot.js) behaves like the Python
 * engine: same intent, same source, confidence within 0.001, same reply for
 * deterministic intents.
 *
 *   python tests/dump_python_results.py > /tmp/py.json
 *   node tests/parity_test.js /tmp/py.json
 */
const fs = require("fs");
const path = require("path");
const { ChatBot, evaluate } = require(path.join(__dirname, "..", "hf_space", "bot.js"));

const data = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "chatbot", "intents.json"), "utf8"));
const expected = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const RANDOM = new Set(["greeting", "how_are_you", "bot_identity", "joke", "thanks", "goodbye", null]);

let failures = 0;
for (const conv of expected) {
  const bot = new ChatBot(data), session = {};
  for (const turn of conv) {
    const r = bot.respond(turn.input, session);
    const problems = [];
    if (r.intent !== turn.intent) problems.push(`intent ${r.intent} != ${turn.intent}`);
    if (r.source !== turn.source) problems.push(`source ${r.source} != ${turn.source}`);
    if (Math.abs(r.confidence - turn.confidence) > 1e-3) problems.push(`confidence ${r.confidence.toFixed(4)} != ${turn.confidence.toFixed(4)}`);
    if (!RANDOM.has(turn.intent) && !turn.source.startsWith("fallback") && r.text !== turn.text) problems.push(`text "${r.text}" != "${turn.text}"`);
    if (problems.length) { failures++; console.log(`FAIL ${JSON.stringify(turn.input)}: ${problems.join("; ")}`); }
  }
}
// arithmetic edge cases (Python reference values)
const math = [["-2**2", -4], ["2**-1", 0.5], ["-7 % 3", 2], ["(3+4)*2", 14], ["2**10", 1024], [".5+1.", 1.5]];
for (const [e, want] of math) if (evaluate(e) !== want) { failures++; console.log(`FAIL evaluate(${e}) = ${evaluate(e)}, want ${want}`); }

const total = expected.reduce((n, c) => n + c.length, 0);
console.log(failures ? `${failures} mismatches` : `PARITY OK: ${total} turns in ${expected.length} conversations + ${math.length} arithmetic checks match the Python engine`);
process.exit(failures ? 1 : 0);
