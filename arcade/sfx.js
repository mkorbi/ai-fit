/* arcade/sfx.js - chiptune blips from WebAudio oscillators, no audio files. Silent until switched on (key S); the
 * audio context starts on that key press, as browsers require. */
const Sfx = (() => {
  let ac = null, on = false;
  function enable(v) {
    on = !!v;
    if (on && !ac) { const AC = window.AudioContext || window.webkitAudioContext; if (AC) ac = new AC(); }
    if (ac && ac.state === 'suspended') ac.resume();
    return on;
  }
  function tone(freq, dur, when = 0, type = 'square', vol = 0.05, slideTo = null) {
    const t0 = ac.currentTime + when;
    const o = ac.createOscillator(), g = ac.createGain();
    o.type = type;
    o.frequency.setValueAtTime(freq, t0);
    if (slideTo) o.frequency.exponentialRampToValueAtTime(slideTo, t0 + dur);
    g.gain.setValueAtTime(vol, t0);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    o.connect(g).connect(ac.destination);
    o.start(t0); o.stop(t0 + dur + 0.02);
  }
  const SOUNDS = {
    blip: () => tone(880, 0.05),
    good: () => { tone(523, 0.07); tone(784, 0.1, 0.07); },
    bad: () => { tone(392, 0.07); tone(262, 0.12, 0.07); },
    error: () => tone(220, 0.35, 0, 'sawtooth', 0.04, 70),
    coin: () => { tone(988, 0.06); tone(1319, 0.16, 0.06); },
    build: () => { for (let i = 0; i < 4; i++) tone(330 + i * 110, 0.05, i * 0.05, 'square', 0.035); },
    start: () => { [523, 659, 784, 1047].forEach((f, i) => tone(f, 0.09, i * 0.08, 'square', 0.04)); },
  };
  function play(name) { if (!on || !ac) return; try { (SOUNDS[name] || SOUNDS.blip)(); } catch (e) { /* audio is decoration */ } }
  return { enable, play, isOn: () => on };
})();
