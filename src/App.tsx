import { useState, useRef, useEffect, useCallback } from 'react';

// ─── Pitch Detection (Autocorrelation) ─────────────────────────────────────
function autoCorrelate(buf, sampleRate) {
  const SIZE = buf.length;
  let rms = 0;
  for (let i = 0; i < SIZE; i++) rms += buf[i] * buf[i];
  rms = Math.sqrt(rms / SIZE);
  if (rms < 0.015) return null;

  let r1 = 0,
    r2 = SIZE - 1;
  const thresh = 0.2;
  for (let i = 0; i < SIZE / 2; i++) {
    if (Math.abs(buf[i]) < thresh) {
      r1 = i;
      break;
    }
  }
  for (let i = 1; i < SIZE / 2; i++) {
    if (Math.abs(buf[SIZE - i]) < thresh) {
      r2 = SIZE - i;
      break;
    }
  }

  const b = buf.slice(r1, r2);
  const c = new Float32Array(b.length).fill(0);
  for (let i = 0; i < b.length; i++)
    for (let j = 0; j < b.length - i; j++) c[i] += b[j] * b[j + i];

  let d = 0;
  while (c[d] > c[d + 1]) d++;
  let maxval = -1,
    maxpos = -1;
  for (let i = d; i < b.length; i++) {
    if (c[i] > maxval) {
      maxval = c[i];
      maxpos = i;
    }
  }
  if (maxpos < 1) return null;

  const x1 = c[maxpos - 1],
    x2 = c[maxpos],
    x3 = maxpos + 1 < c.length ? c[maxpos + 1] : x2;
  const a = (x1 + x3 - 2 * x2) / 2;
  const b2 = (x3 - x1) / 2;
  const T0 = a ? maxpos - b2 / (2 * a) : maxpos;
  return sampleRate / T0;
}

// ─── Note / Swara Mapping ──────────────────────────────────────────────────
const SA_FREQS = {
  C: 261.63,
  'C#': 277.18,
  D: 293.66,
  'D#': 311.13,
  E: 329.63,
  F: 349.23,
  'F#': 369.99,
  G: 392.0,
  'G#': 415.3,
  A: 440.0,
  'A#': 466.16,
  B: 493.88,
};

const SWARAS = [
  { id: 'S', label: 'सा', name: 'Sa', semitone: 0 },
  { id: 'r', label: 'रे॒', name: 'Komal Re', semitone: 1 },
  { id: 'R', label: 'रे', name: 'Re', semitone: 2 },
  { id: 'g', label: 'ग॒', name: 'Komal Ga', semitone: 3 },
  { id: 'G', label: 'ग', name: 'Ga', semitone: 4 },
  { id: 'M', label: 'म', name: 'Ma', semitone: 5 },
  { id: 'm', label: 'म॑', name: 'Tivra Ma', semitone: 6 },
  { id: 'P', label: 'प', name: 'Pa', semitone: 7 },
  { id: 'd', label: 'ध॒', name: 'Komal Dha', semitone: 8 },
  { id: 'D', label: 'ध', name: 'Dha', semitone: 9 },
  { id: 'n', label: 'नि॒', name: 'Komal Ni', semitone: 10 },
  { id: 'N', label: 'नि', name: 'Ni', semitone: 11 },
];

function freqToSwara(freq, saFreq) {
  if (!freq || freq < 60 || freq > 2000) return null;
  // Find closest octave of Sa
  let saRef = saFreq;
  while (saRef * 2 < freq) saRef *= 2;
  while (saRef > freq * 1.5) saRef /= 2;
  const semitones = Math.round(12 * Math.log2(freq / saRef));
  const norm = ((semitones % 12) + 12) % 12;
  return SWARAS.find((s) => s.semitone === norm) || null;
}

// ─── Component ─────────────────────────────────────────────────────────────
const STATES = {
  IDLE: 'idle',
  LISTENING: 'listening',
  ANALYZING: 'analyzing',
  RESULT: 'result',
};

export default function RaagaShazam() {
  const [appState, setAppState] = useState(STATES.IDLE);
  const [sa, setSa] = useState('C');
  const [detectedSwaras, setDetectedSwaras] = useState([]); // ordered unique swaras
  const [currentSwara, setCurrentSwara] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const [bars, setBars] = useState(Array(32).fill(2));
  const [permError, setPermError] = useState(false);

  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);
  const timerRef = useRef(null);
  const swaraHistoryRef = useRef([]);
  const detectedSetRef = useRef(new Set());
  const elapsedRef = useRef(0);
  const MAX_LISTEN = 10;

  const stopListening = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    clearInterval(timerRef.current);
    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch {}
    }
    if (streamRef.current)
      streamRef.current.getTracks().forEach((t) => t.stop());
    if (audioCtxRef.current) {
      try {
        audioCtxRef.current.close();
      } catch {}
    }
    audioCtxRef.current = null;
    setBars(Array(32).fill(2));
    setCurrentSwara(null);
  }, []);

  const analyze = useCallback(async () => {
    stopListening();
    const detected = [...detectedSetRef.current];
    if (detected.length < 2) {
      setError('Not enough notes detected. Try playing/singing more clearly.');
      setAppState(STATES.IDLE);
      return;
    }
    setAppState(STATES.ANALYZING);
    try {
      const swaraList = detected.map((id) => {
        const s = SWARAS.find((sw) => sw.id === id);
        return `${s.name} (${s.label})`;
      });
      const prompt = `I recorded a melody in Indian classical music. The Sa (tonic) is set to ${sa}. The following swaras were detected in this session: ${swaraList.join(
        ', '
      )}.

Based on this swara set, identify the most likely Indian classical raaga. Consider Hindustani tradition primarily.

Respond ONLY with this exact JSON, no extra text or markdown:
{
  "primaryRaaga": "Raaga name",
  "thaat": "Thaat name",
  "confidence": "High/Medium/Low",
  "timeOfDay": "Morning/Afternoon/Evening/Night/All times",
  "season": "Spring/Summer/Monsoon/Autumn/Winter/All seasons",
  "mood": "Peaceful/Romantic/Devotional/Meditative/Energetic/Melancholic/Heroic/Serene",
  "moodDescription": "One evocative line capturing the emotional essence",
  "aaroha": "S R G M P D N S'",
  "avaroha": "S' N D P M G R S",
  "vadi": "Most important swara",
  "samvadi": "Second most important swara",
  "detectedMatch": ["List only the detected swaras that matched this raaga"],
  "missingForPerfectMatch": ["Key swaras of this raaga not detected, or empty array"],
  "alternateRaagas": ["Alternate raaga 1", "Alternate raaga 2"],
  "famousRaagaIn": "Famous compositions or artists known for this raaga",
  "shortDescription": "2-3 sentences on this raaga's personality and history"
}`;
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 1000,
          messages: [{ role: 'user', content: prompt }],
        }),
      });
      const data = await response.json();
      const text = data.content?.map((c) => c.text || '').join('') || '';
      const parsed = JSON.parse(text.replace(/```json|```/g, '').trim());
      setResult(parsed);
      setAppState(STATES.RESULT);
    } catch {
      setError('Could not identify the raaga. Please try again.');
      setAppState(STATES.IDLE);
    }
  }, [sa, stopListening]);

  const startListening = async () => {
    setError(null);
    setResult(null);
    setDetectedSwaras([]);
    setCurrentSwara(null);
    setElapsed(0);
    elapsedRef.current = 0;
    detectedSetRef.current = new Set();
    swaraHistoryRef.current = [];
    setPermError(false);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: false,
      });
      streamRef.current = stream;
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      audioCtxRef.current = ctx;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      analyserRef.current = analyser;
      const source = ctx.createMediaStreamSource(stream);
      sourceRef.current = source;
      source.connect(analyser);
      setAppState(STATES.LISTENING);

      const buf = new Float32Array(analyser.fftSize);
      const freqBuf = new Uint8Array(analyser.frequencyBinCount);
      const saFreq = SA_FREQS[sa];

      const tick = () => {
        analyser.getFloatTimeDomainData(buf);
        analyser.getByteFrequencyData(freqBuf);

        // Waveform bars
        const step = Math.floor(freqBuf.length / 32);
        setBars(
          Array.from({ length: 32 }, (_, i) => {
            const val = freqBuf[i * step] || 0;
            return Math.max(2, (val / 255) * 80);
          })
        );

        // Pitch detect
        const freq = autoCorrelate(buf, ctx.sampleRate);
        const swara = freq ? freqToSwara(freq, saFreq) : null;

        if (swara) {
          swaraHistoryRef.current.push(swara.id);
          if (swaraHistoryRef.current.length > 6)
            swaraHistoryRef.current.shift();
          // Majority vote over last 6 frames
          const counts = {};
          swaraHistoryRef.current.forEach((id) => {
            counts[id] = (counts[id] || 0) + 1;
          });
          const stable = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
          if (stable && stable[1] >= 4) {
            const stableSwara = SWARAS.find((s) => s.id === stable[0]);
            setCurrentSwara(stableSwara);
            if (!detectedSetRef.current.has(stable[0])) {
              detectedSetRef.current.add(stable[0]);
              setDetectedSwaras((prev) => [...prev, stable[0]]);
            }
          }
        } else {
          setCurrentSwara(null);
        }
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);

      timerRef.current = setInterval(() => {
        elapsedRef.current += 1;
        setElapsed(elapsedRef.current);
        if (elapsedRef.current >= MAX_LISTEN) {
          clearInterval(timerRef.current);
          analyze();
        }
      }, 1000);
    } catch (e) {
      setPermError(true);
      setError(
        'Microphone access denied. Please allow mic permissions and try again.'
      );
    }
  };

  const reset = () => {
    stopListening();
    setAppState(STATES.IDLE);
    setDetectedSwaras([]);
    setCurrentSwara(null);
    setResult(null);
    setError(null);
    setElapsed(0);
    detectedSetRef.current = new Set();
    swaraHistoryRef.current = [];
  };

  useEffect(() => () => stopListening(), [stopListening]);

  const progress = Math.min(elapsed / MAX_LISTEN, 1);
  const circumference = 2 * Math.PI * 68;

  return (
    <div
      style={{
        minHeight: '100vh',
        background: '#060810',
        fontFamily: "'DM Sans', 'Helvetica Neue', sans-serif",
        color: '#e8eaf0',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'flex-start',
        padding: '0 20px 60px',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <link
        href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&family=Noto+Serif+Devanagari:wght@400;600&display=swap"
        rel="stylesheet"
      />

      <style>{`
        @keyframes ripple {
          0% { transform: scale(1); opacity: 0.6; }
          100% { transform: scale(2.5); opacity: 0; }
        }
        @keyframes pulse-ring {
          0% { transform: scale(0.95); opacity: 0.8; }
          50% { transform: scale(1.05); opacity: 0.4; }
          100% { transform: scale(0.95); opacity: 0.8; }
        }
        @keyframes fadein { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
        @keyframes popin { from { opacity:0; transform:scale(0.7); } to { opacity:1; transform:scale(1); } }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes glow { 0%,100% { box-shadow: 0 0 30px rgba(255,140,50,0.3); } 50% { box-shadow: 0 0 60px rgba(255,140,50,0.6), 0 0 100px rgba(255,140,50,0.2); } }
        @keyframes idle-glow { 0%,100% { box-shadow: 0 0 20px rgba(100,140,255,0.2); } 50% { box-shadow: 0 0 40px rgba(100,140,255,0.4); } }
        .swara-chip { animation: popin 0.25s cubic-bezier(.34,1.56,.64,1); }
        .btn-listen { transition: transform 0.15s; }
        .btn-listen:hover { transform: scale(1.04); }
        .btn-listen:active { transform: scale(0.97); }
        .sa-option { transition: all 0.15s; cursor: pointer; }
        .sa-option:hover { border-color: rgba(255,140,50,0.5) !important; }
      `}</style>

      {/* Background aurora */}
      <div
        style={{
          position: 'fixed',
          inset: 0,
          zIndex: 0,
          pointerEvents: 'none',
        }}
      >
        <div
          style={{
            position: 'absolute',
            top: '10%',
            left: '50%',
            transform: 'translateX(-50%)',
            width: '70vw',
            height: '50vw',
            maxWidth: 600,
            maxHeight: 400,
            background:
              'radial-gradient(ellipse, rgba(80,100,255,0.06) 0%, transparent 70%)',
            borderRadius: '50%',
          }}
        />
        <div
          style={{
            position: 'absolute',
            bottom: '0%',
            left: '20%',
            width: '40vw',
            height: '30vw',
            background:
              'radial-gradient(ellipse, rgba(255,100,60,0.04) 0%, transparent 70%)',
          }}
        />
      </div>

      <div
        style={{
          position: 'relative',
          zIndex: 1,
          width: '100%',
          maxWidth: 480,
        }}
      >
        {/* Header */}
        <div style={{ textAlign: 'center', padding: '44px 0 32px' }}>
          <div
            style={{
              fontSize: '0.72rem',
              letterSpacing: '0.22em',
              color: '#4a5070',
              textTransform: 'uppercase',
              marginBottom: 8,
            }}
          >
            ◈ RAAGA SHRAVAN ◈
          </div>
          <h1
            style={{
              fontFamily: "'DM Serif Display', serif",
              fontSize: 'clamp(1.6rem, 5vw, 2.2rem)',
              fontWeight: 400,
              color: '#e8eaf0',
              margin: 0,
              letterSpacing: '-0.01em',
            }}
          >
            Listen & Identify
          </h1>
          <p
            style={{
              color: '#3a4060',
              fontSize: '0.82rem',
              margin: '8px 0 0',
              letterSpacing: '0.04em',
            }}
          >
            Play or sing — we'll detect the raaga
          </p>
        </div>

        {/* Sa Selector */}
        {appState === STATES.IDLE && (
          <div style={{ animation: 'fadein 0.4s ease', marginBottom: 36 }}>
            <p
              style={{
                textAlign: 'center',
                color: '#3a4060',
                fontSize: '0.72rem',
                letterSpacing: '0.12em',
                textTransform: 'uppercase',
                marginBottom: 12,
              }}
            >
              Set your Sa (Tonic)
            </p>
            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: 6,
                justifyContent: 'center',
              }}
            >
              {Object.keys(SA_FREQS).map((note) => (
                <button
                  key={note}
                  className="sa-option"
                  onClick={() => setSa(note)}
                  style={{
                    padding: '6px 13px',
                    background:
                      sa === note ? 'rgba(255,140,50,0.12)' : 'transparent',
                    border: `1px solid ${
                      sa === note ? 'rgba(255,140,50,0.6)' : '#1a1e30'
                    }`,
                    borderRadius: 6,
                    cursor: 'pointer',
                    color: sa === note ? '#ff8c32' : '#3a4060',
                    fontSize: '0.8rem',
                    fontWeight: sa === note ? 600 : 400,
                    fontFamily: "'DM Sans', sans-serif",
                  }}
                >
                  {note}
                </button>
              ))}
            </div>
            <p
              style={{
                textAlign: 'center',
                color: '#252840',
                fontSize: '0.72rem',
                marginTop: 8,
              }}
            >
              Not sure? C works for most instruments
            </p>
          </div>
        )}

        {/* Main Listen Button */}
        {(appState === STATES.IDLE || appState === STATES.LISTENING) && (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              marginBottom: 32,
            }}
          >
            {/* Circular progress + button */}
            <div style={{ position: 'relative', width: 160, height: 160 }}>
              {/* Ripples when listening */}
              {appState === STATES.LISTENING &&
                [1, 2, 3].map((i) => (
                  <div
                    key={i}
                    style={{
                      position: 'absolute',
                      inset: 0,
                      borderRadius: '50%',
                      border: '1px solid rgba(255,140,50,0.3)',
                      animation: `ripple ${1.8 + i * 0.5}s ease-out infinite`,
                      animationDelay: `${i * 0.4}s`,
                    }}
                  />
                ))}
              {/* SVG ring */}
              <svg
                width="160"
                height="160"
                style={{
                  position: 'absolute',
                  inset: 0,
                  transform: 'rotate(-90deg)',
                }}
              >
                <circle
                  cx="80"
                  cy="80"
                  r="68"
                  fill="none"
                  stroke="#0e1020"
                  strokeWidth="4"
                />
                {appState === STATES.LISTENING && (
                  <circle
                    cx="80"
                    cy="80"
                    r="68"
                    fill="none"
                    stroke="#ff8c32"
                    strokeWidth="4"
                    strokeDasharray={circumference}
                    strokeDashoffset={circumference * (1 - progress)}
                    strokeLinecap="round"
                    style={{ transition: 'stroke-dashoffset 0.9s linear' }}
                  />
                )}
              </svg>
              {/* Center button */}
              <button
                className="btn-listen"
                onClick={appState === STATES.IDLE ? startListening : analyze}
                style={{
                  position: 'absolute',
                  inset: 12,
                  borderRadius: '50%',
                  border: 'none',
                  background:
                    appState === STATES.LISTENING
                      ? 'radial-gradient(circle, #ff8c32 0%, #c85a10 100%)'
                      : 'radial-gradient(circle, #1e2240 0%, #0e1020 100%)',
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 4,
                  animation:
                    appState === STATES.LISTENING
                      ? 'glow 2s ease-in-out infinite'
                      : 'idle-glow 3s ease-in-out infinite',
                }}
              >
                {appState === STATES.LISTENING ? (
                  <>
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                      <rect
                        x="6"
                        y="6"
                        width="12"
                        height="12"
                        rx="2"
                        fill="white"
                      />
                    </svg>
                    <span
                      style={{
                        color: 'white',
                        fontSize: '0.65rem',
                        letterSpacing: '0.1em',
                        opacity: 0.9,
                      }}
                    >
                      STOP
                    </span>
                  </>
                ) : (
                  <>
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
                      <path
                        d="M12 1a4 4 0 0 1 4 4v7a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4z"
                        fill="#6479ff"
                        opacity="0.9"
                      />
                      <path
                        d="M19 10v2a7 7 0 0 1-14 0v-2"
                        stroke="#6479ff"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                      />
                      <line
                        x1="12"
                        y1="19"
                        x2="12"
                        y2="23"
                        stroke="#6479ff"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                      />
                    </svg>
                    <span
                      style={{
                        color: '#6479ff',
                        fontSize: '0.62rem',
                        letterSpacing: '0.1em',
                      }}
                    >
                      TAP TO LISTEN
                    </span>
                  </>
                )}
              </button>
            </div>

            {appState === STATES.LISTENING && (
              <div
                style={{
                  marginTop: 14,
                  color: '#ff8c32',
                  fontSize: '0.78rem',
                  letterSpacing: '0.06em',
                }}
              >
                Listening… {MAX_LISTEN - elapsed}s left
              </div>
            )}
          </div>
        )}

        {/* Waveform bars */}
        {appState === STATES.LISTENING && (
          <div
            style={{
              display: 'flex',
              alignItems: 'flex-end',
              justifyContent: 'center',
              gap: 3,
              height: 80,
              marginBottom: 28,
              animation: 'fadein 0.3s ease',
            }}
          >
            {bars.map((h, i) => (
              <div
                key={i}
                style={{
                  width: 5,
                  borderRadius: 3,
                  height: h,
                  background: `linear-gradient(to top, #ff8c32, #ffb06080)`,
                  transition: 'height 0.08s ease',
                  opacity: 0.5 + (h / 80) * 0.5,
                }}
              />
            ))}
          </div>
        )}

        {/* Current detected swara */}
        {appState === STATES.LISTENING && (
          <div
            style={{
              textAlign: 'center',
              marginBottom: 24,
              minHeight: 60,
            }}
          >
            {currentSwara ? (
              <div
                key={currentSwara.id}
                style={{ animation: 'fadein 0.15s ease' }}
              >
                <div
                  style={{
                    fontFamily: "'Noto Serif Devanagari', serif",
                    fontSize: '2.2rem',
                    color: '#ff8c32',
                    lineHeight: 1,
                  }}
                >
                  {currentSwara.label}
                </div>
                <div
                  style={{ color: '#5a6080', fontSize: '0.8rem', marginTop: 4 }}
                >
                  {currentSwara.name}
                </div>
              </div>
            ) : (
              <div
                style={{
                  color: '#252840',
                  fontSize: '0.8rem',
                  fontStyle: 'italic',
                  paddingTop: 12,
                }}
              >
                Waiting for a note…
              </div>
            )}
          </div>
        )}

        {/* Detected swara chips */}
        {appState === STATES.LISTENING && detectedSwaras.length > 0 && (
          <div style={{ animation: 'fadein 0.3s ease', marginBottom: 28 }}>
            <p
              style={{
                textAlign: 'center',
                color: '#252840',
                fontSize: '0.7rem',
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
                marginBottom: 12,
              }}
            >
              Detected so far
            </p>
            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: 8,
                justifyContent: 'center',
              }}
            >
              {detectedSwaras.map((id) => {
                const sw = SWARAS.find((s) => s.id === id);
                return (
                  <div
                    key={id}
                    className="swara-chip"
                    style={{
                      padding: '6px 14px',
                      background: 'rgba(255,140,50,0.1)',
                      border: '1px solid rgba(255,140,50,0.3)',
                      borderRadius: 20,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                    }}
                  >
                    <span
                      style={{
                        fontFamily: "'Noto Serif Devanagari', serif",
                        fontSize: '1rem',
                        color: '#ff8c32',
                      }}
                    >
                      {sw.label}
                    </span>
                    <span style={{ fontSize: '0.72rem', color: '#5a6080' }}>
                      {sw.name}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Analyzing */}
        {appState === STATES.ANALYZING && (
          <div
            style={{
              textAlign: 'center',
              padding: '60px 0',
              animation: 'fadein 0.4s ease',
            }}
          >
            <div
              style={{
                width: 56,
                height: 56,
                border: '2px solid #0e1020',
                borderTop: '2px solid #ff8c32',
                borderRadius: '50%',
                margin: '0 auto 20px',
                animation: 'spin 0.8s linear infinite',
              }}
            />
            <div
              style={{
                color: '#4a5070',
                fontSize: '0.88rem',
                fontStyle: 'italic',
              }}
            >
              Consulting the shastra…
            </div>
            <div
              style={{ color: '#252840', fontSize: '0.75rem', marginTop: 6 }}
            >
              Identified {detectedSwaras.length} swaras with Sa={sa}
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div
            style={{
              background: 'rgba(180,50,50,0.08)',
              border: '1px solid rgba(180,50,50,0.2)',
              borderRadius: 10,
              padding: '14px 18px',
              marginBottom: 20,
              color: '#c05050',
              fontSize: '0.85rem',
              textAlign: 'center',
              animation: 'fadein 0.3s ease',
            }}
          >
            {error}
            {permError && (
              <div
                style={{ fontSize: '0.75rem', marginTop: 6, color: '#7a3030' }}
              >
                Go to browser settings → Site permissions → Microphone → Allow
              </div>
            )}
          </div>
        )}

        {/* ─── RESULT ──────────────────────────────────────────────── */}
        {appState === STATES.RESULT && result && (
          <div style={{ animation: 'fadein 0.5s ease' }}>
            {/* Hero card */}
            <div
              style={{
                background:
                  'linear-gradient(135deg, rgba(255,140,50,0.08) 0%, rgba(255,100,50,0.04) 100%)',
                border: '1px solid rgba(255,140,50,0.15)',
                borderRadius: 20,
                padding: '30px 28px',
                marginBottom: 12,
                position: 'relative',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  position: 'absolute',
                  top: -40,
                  right: -40,
                  width: 160,
                  height: 160,
                  background:
                    'radial-gradient(circle, rgba(255,140,50,0.08) 0%, transparent 70%)',
                  borderRadius: '50%',
                }}
              />
              <div
                style={{
                  fontSize: '0.65rem',
                  letterSpacing: '0.16em',
                  color: '#4a5070',
                  textTransform: 'uppercase',
                  marginBottom: 12,
                }}
              >
                Raaga Identified
              </div>
              <div
                style={{
                  fontFamily: "'DM Serif Display', serif",
                  fontSize: 'clamp(2rem, 7vw, 2.8rem)',
                  color: '#f0e8d8',
                  lineHeight: 1,
                  marginBottom: 6,
                }}
              >
                {result.primaryRaaga}
              </div>
              <div
                style={{
                  color: '#4a5070',
                  fontSize: '0.82rem',
                  marginBottom: 18,
                }}
              >
                {result.thaat} Thaat · {result.timeOfDay}
              </div>

              {/* Confidence pill */}
              <div
                style={{
                  display: 'flex',
                  gap: 8,
                  flexWrap: 'wrap',
                  marginBottom: 18,
                }}
              >
                {[
                  {
                    label: result.confidence + ' confidence',
                    color:
                      result.confidence === 'High'
                        ? '#3a8c5a'
                        : result.confidence === 'Medium'
                        ? '#c87820'
                        : '#8c3a3a',
                  },
                  { label: '🌿 ' + result.season, color: '#4a5070' },
                  { label: result.mood, color: '#4a5070' },
                ].map((tag, i) => (
                  <span
                    key={i}
                    style={{
                      padding: '4px 11px',
                      borderRadius: 20,
                      fontSize: '0.72rem',
                      border: `1px solid ${tag.color}55`,
                      color: tag.color,
                      background: `${tag.color}11`,
                    }}
                  >
                    {tag.label}
                  </span>
                ))}
              </div>

              <p
                style={{
                  fontFamily: "'DM Serif Display', serif",
                  fontStyle: 'italic',
                  color: '#8090b0',
                  fontSize: '0.95rem',
                  lineHeight: 1.65,
                  margin: 0,
                }}
              >
                "{result.moodDescription}"
              </p>
            </div>

            {/* Scale */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: 10,
                marginBottom: 10,
              }}
            >
              {[
                { label: 'Aaroha ↑', val: result.aaroha },
                { label: 'Avaroha ↓', val: result.avaroha },
              ].map((item) => (
                <div
                  key={item.label}
                  style={{
                    background: 'rgba(255,255,255,0.02)',
                    border: '1px solid #0e1020',
                    borderRadius: 12,
                    padding: '14px 16px',
                  }}
                >
                  <div
                    style={{
                      color: '#2a3050',
                      fontSize: '0.68rem',
                      letterSpacing: '0.1em',
                      textTransform: 'uppercase',
                      marginBottom: 6,
                    }}
                  >
                    {item.label}
                  </div>
                  <div
                    style={{
                      color: '#8090b0',
                      fontFamily: 'monospace',
                      fontSize: '0.85rem',
                    }}
                  >
                    {item.val}
                  </div>
                </div>
              ))}
            </div>

            {/* Vadi / Samvadi */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: 10,
                marginBottom: 10,
              }}
            >
              {[
                { label: 'Vadi (King Note)', val: result.vadi },
                { label: 'Samvadi (Minister)', val: result.samvadi },
              ].map((item) => (
                <div
                  key={item.label}
                  style={{
                    background: 'rgba(255,255,255,0.02)',
                    border: '1px solid #0e1020',
                    borderRadius: 12,
                    padding: '14px 16px',
                    textAlign: 'center',
                  }}
                >
                  <div
                    style={{
                      color: '#2a3050',
                      fontSize: '0.68rem',
                      letterSpacing: '0.1em',
                      textTransform: 'uppercase',
                      marginBottom: 6,
                    }}
                  >
                    {item.label}
                  </div>
                  <div
                    style={{
                      color: '#ff8c32',
                      fontSize: '1rem',
                      fontStyle: 'italic',
                    }}
                  >
                    {item.val}
                  </div>
                </div>
              ))}
            </div>

            {/* Match quality */}
            {(result.detectedMatch?.length > 0 ||
              result.missingForPerfectMatch?.length > 0) && (
              <div
                style={{
                  background: 'rgba(255,255,255,0.02)',
                  border: '1px solid #0e1020',
                  borderRadius: 12,
                  padding: '16px 18px',
                  marginBottom: 10,
                }}
              >
                <div
                  style={{
                    color: '#2a3050',
                    fontSize: '0.68rem',
                    letterSpacing: '0.1em',
                    textTransform: 'uppercase',
                    marginBottom: 10,
                  }}
                >
                  Detection Match
                </div>
                {result.detectedMatch?.length > 0 && (
                  <div style={{ marginBottom: 8 }}>
                    <span style={{ color: '#3a8c5a', fontSize: '0.72rem' }}>
                      ✓ Matched:{' '}
                    </span>
                    <span style={{ color: '#5a6080', fontSize: '0.78rem' }}>
                      {result.detectedMatch.join(', ')}
                    </span>
                  </div>
                )}
                {result.missingForPerfectMatch?.length > 0 && (
                  <div>
                    <span style={{ color: '#8c6a30', fontSize: '0.72rem' }}>
                      ○ Not heard:{' '}
                    </span>
                    <span style={{ color: '#3a4060', fontSize: '0.78rem' }}>
                      {result.missingForPerfectMatch.join(', ')}
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* About */}
            <div
              style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid #0e1020',
                borderRadius: 12,
                padding: '16px 18px',
                marginBottom: 10,
              }}
            >
              <div
                style={{
                  color: '#2a3050',
                  fontSize: '0.68rem',
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  marginBottom: 8,
                }}
              >
                About this Raaga
              </div>
              <p
                style={{
                  color: '#4a5070',
                  fontSize: '0.85rem',
                  lineHeight: 1.75,
                  margin: 0,
                  fontStyle: 'italic',
                }}
              >
                {result.shortDescription}
              </p>
              {result.famousRaagaIn && (
                <div
                  style={{
                    marginTop: 10,
                    color: '#3a4060',
                    fontSize: '0.78rem',
                  }}
                >
                  🎵 {result.famousRaagaIn}
                </div>
              )}
            </div>

            {/* Alternates */}
            {result.alternateRaagas?.filter(Boolean).length > 0 && (
              <div
                style={{
                  background: 'rgba(255,255,255,0.02)',
                  border: '1px solid #0e1020',
                  borderRadius: 12,
                  padding: '14px 18px',
                  marginBottom: 20,
                }}
              >
                <div
                  style={{
                    color: '#2a3050',
                    fontSize: '0.68rem',
                    letterSpacing: '0.1em',
                    textTransform: 'uppercase',
                    marginBottom: 8,
                  }}
                >
                  Could also be
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {result.alternateRaagas.filter(Boolean).map((r, i) => (
                    <span
                      key={i}
                      style={{
                        padding: '5px 13px',
                        border: '1px solid #0e1020',
                        borderRadius: 20,
                        fontSize: '0.8rem',
                        color: '#3a4060',
                      }}
                    >
                      {r}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* CTA */}
            <button
              onClick={reset}
              style={{
                width: '100%',
                padding: '14px',
                background: 'rgba(255,140,50,0.08)',
                border: '1px solid rgba(255,140,50,0.2)',
                borderRadius: 12,
                cursor: 'pointer',
                color: '#ff8c32',
                fontFamily: "'DM Sans', sans-serif",
                fontSize: '0.9rem',
                fontWeight: 500,
                letterSpacing: '0.04em',
                transition: 'all 0.18s',
              }}
            >
              Listen Again
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
