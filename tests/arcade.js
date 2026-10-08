#!/usr/bin/env node
/* arcade.js — every mission must be losable from its start state and winnable with its intended move. */
const path = require('path');
const A = require(path.join(__dirname, '..', 'game.js'));
let fail = 0;
for (const lv of A.LEVELS) {
  if (!lv.goal) continue;
  const s0 = A.stateFor(lv), e0 = A.evaluateState(s0, lv);
  const s1 = Object.assign({}, s0, lv.solve), e1 = A.evaluateState(s1, lv);
  const ok = !e0.ok && e1.ok;
  if (!ok) fail++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${lv.id.padEnd(8)} start → ${e0.ok ? 'WINS (should not)' : 'loses: ' + e0.reasons[0]}   | solve → ${e1.ok ? 'wins' + (e1.stars ? ' ' + '★'.repeat(e1.stars) : '') : 'LOSES: ' + e1.reasons[0]}`);
}
console.log(`\n${fail} mission${fail === 1 ? '' : 's'} broken.`);
process.exit(fail ? 1 : 0);
