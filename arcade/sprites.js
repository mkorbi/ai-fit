/* arcade/sprites.js - pixel art for the arcade world: people, desks, icons, and a 3×5 bitmap font.
 * Art is drawn as strings; each character is a palette key ('.' is transparent). bake() turns a sprite into an offscreen
 * canvas for a given color map, so the world draws whole sprites with drawImage at integer positions.
 */
const Sprites = (() => {
  /* People are 5 wide and 9 tall. H hair, F skin, S shirt, P trousers, B shoes. Two frames for walking and waiting. */
  const PERSON = [
    ['.HHH.', '.FFF.', '.FFF.', 'SSSSS', 'FSSSF', '.SSS.', '.PPP.', '.P.P.', '.B.B.'],
    ['.HHH.', '.FFF.', '.FFF.', 'SSSSS', 'FSSSF', '.SSS.', '.PPP.', '.P.P.', 'B...B'],
  ];
  /* A person at a desk, 8 wide and 8 tall (a desk slot is 9 wide, so neighbours keep a gap): D desk, d desk shadow,
   * M monitor frame, G screen glow, g screen off. Two typing frames. */
  const SEATED = [
    ['.HHH....', '.FFF....', '.FFF..MM', 'SSSSS.MG', 'FSSSF.MG', 'DDDDDDDD', 'dD....Dd', '.D....D.'],
    ['.HHH....', '.FFF....', '.FFF..MM', 'SSSSS.MG', '.SSS..MG', 'DFDDFDDD', 'dD....Dd', '.D....D.'],
  ];
  /* Head down on the desk: an idle session still holding GPU memory. */
  const SLEEPING = ['........', '........', '......MM', '.HHH..Mg', 'SFFFS.Mg', 'DDDDDDDD', 'dD....Dd', '.D....D.'];
  const DESK = ['........', '........', '......MM', '......Mg', '......Mg', 'DDDDDDDD', 'dD....Dd', '.D....D.'];
  const ICONS = {
    zzz: ['ZZZ', '..Z', '.Z.', 'ZZZ'],
    sweat: ['.W', 'WW', 'WW'],
    hourglass: ['YYY', '.Y.', 'YYY'],
    suitcase: ['.K.', 'KKK', 'KKK'],
    bang: ['R', 'R', 'R', '.', 'R'],
    cross: ['R.R', '.R.', 'R.R'],
  };

  /* 3×5 font: uppercase, digits and the punctuation the stage needs. Each glyph is 5 rows of 3 bits. */
  const G = {
    A: '010101111101101', B: '110101110101110', C: '011100100100011', D: '110101101101110', E: '111100110100111', F: '111100110100100',
    G: '011100101101011', H: '101101111101101', I: '111010010010111', J: '001001001101010', K: '101101110101101', L: '100100100100111',
    M: '101111111101101', N: '110101101101101', O: '010101101101010', P: '110101110100100', Q: '010101101110011', R: '110101110101101',
    S: '011100010001110', T: '111010010010010', U: '101101101101111', V: '101101101101010', W: '101101111111101', X: '101101010101101',
    Y: '101101010010010', Z: '111001010100111',
    0: '111101101101111', 1: '010110010010111', 2: '110001010100111', 3: '110001010001110', 4: '101101111001001', 5: '111100110001110',
    6: '011100111101111', 7: '111001010010010', 8: '111101111101111', 9: '111101111001110',
    ' ': '000000000000000', '.': '000000000000010', ',': '000000000010100', ':': '000010000010000', '-': '000000111000000', '+': '000010111010000',
    '/': '001001010100100', '%': '101001010100101', '$': '011110010011110', '(': '010100100100010', ')': '010001001001010', '!': '010010010000010',
    '?': '110001010000010', '=': '000111000111000', '<': '001010100010001', '>': '100010001010100', '×': '000101010101000', '~': '000011110000000',
    "'": '010010000000000', '#': '101111101111101', '@': '010101111100011', '_': '000000000000111', '*': '000101010101000', '→': '100010001010100',
  };
  const GLYPH_W = 3, GLYPH_H = 5, ADVANCE = 4;
  const textWidth = (s) => (s.length ? s.length * ADVANCE - 1 : 0);
  /* Draw uppercase text at integer (x, y) in the given color. Characters without a glyph are skipped. */
  function text(ctx, str, x, y, color) {
    ctx.fillStyle = color;
    let cx = Math.round(x);
    for (const ch of String(str).toUpperCase()) {
      const g = G[ch];
      if (g) for (let i = 0; i < 15; i++) if (g[i] === '1') ctx.fillRect(cx + (i % 3), Math.round(y) + Math.floor(i / 3), 1, 1);
      cx += ADVANCE;
    }
  }

  /* An offscreen canvas with the art drawn in the colors of `map` (palette key -> CSS color). */
  function bake(rows, map) {
    const w = rows[0].length, h = rows.length;
    const c = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : Object.assign(document.createElement('canvas'), { width: w, height: h });
    const ctx = c.getContext('2d');
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const k = rows[y][x];
      if (k === '.' || !map[k]) continue;
      ctx.fillStyle = map[k];
      ctx.fillRect(x, y, 1, 1);
    }
    return c;
  }

  /* Deterministic variety: skin, hair and shirt colors per person id. */
  function variant(id, pal) {
    const pick = (list, salt) => list[(Math.imul(id * 31 + salt, 2654435761) >>> 0) % list.length];
    return { skin: pick(pal.skin, 7), hair: pick(pal.hair, 13), shirt: pick(pal.shirt, 29) };
  }

  return { PERSON, SEATED, SLEEPING, DESK, ICONS, GLYPHS: G, GLYPH_W, GLYPH_H, ADVANCE, text, textWidth, bake, variant };
})();

if (typeof module !== 'undefined') module.exports = Sprites;
