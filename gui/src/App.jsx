import { useState, useEffect, useRef } from "react";

const UNIVERSE_META = {
  star_wars:    { emoji: "⭐", color: "#FFE81F", bg: "#0d0d1a" },
  harry_potter: { emoji: "⚡", color: "#AE0001", bg: "#1a0a0a" },
  lotr:         { emoji: "💍", color: "#C0962C", bg: "#0a110a" },
};

const API = "";  // proxied by Vite; change to "http://localhost:8000" if running standalone

export default function App() {
  const [universes, setUniverses]     = useState([]);
  const [selected, setSelected]       = useState(null);
  const [prompt, setPrompt]           = useState("");
  const [history, setHistory]         = useState([]);   // [{id, universe, prompt, result}]
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const [maxTokens, setMaxTokens]     = useState(256);
  const [temperature, setTemperature] = useState(0.9);
  const [useRag, setUseRag]           = useState(true);
  const bottomRef = useRef(null);

  // Load universe list on mount
  useEffect(() => {
    fetch(`${API}/universes`)
      .then(r => r.json())
      .then(data => {
        setUniverses(data);
        const first = data.find(u => u.available);
        if (first) setSelected(first.id);
      })
      .catch(() => setError("Could not reach the LoreForge server. Is server.py running?"));
  }, []);

  // Scroll to bottom when history changes
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  async function handleGenerate() {
    if (!prompt.trim() || !selected || loading) return;
    setLoading(true);
    setError(null);

    const entry = { id: Date.now(), universe: selected, prompt: prompt.trim(), result: null };
    setHistory(h => [...h, entry]);
    setPrompt("");

    try {
      const res = await fetch(`${API}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          universe: selected,
          prompt: entry.prompt,
          max_new_tokens: maxTokens,
          temperature: temperature,
          top_k: 50,
          use_rag: useRag,
        }),
      });
      if (!res.ok) {
        const detail = (await res.json()).detail ?? "Unknown error";
        throw new Error(detail);
      }
      const result = await res.json();
      setHistory(h => h.map(e => e.id === entry.id ? { ...e, result } : e));
    } catch (err) {
      setError(err.message);
      setHistory(h => h.map(e => e.id === entry.id ? { ...e, result: { error: err.message } } : e));
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleGenerate();
  }

  const meta = selected ? (UNIVERSE_META[selected] ?? UNIVERSE_META.star_wars) : { color: "#888", bg: "#111" };

  return (
    <div style={styles.root}>
      {/* ── Header ── */}
      <header style={styles.header}>
        <h1 style={styles.title}>LoreForge</h1>
        <p style={styles.subtitle}>Multi-Universe Lore-Faithful Story Generation</p>
      </header>

      {/* ── Universe Picker ── */}
      <section style={styles.pickerSection}>
        <div style={styles.pickerRow}>
          {universes.map(u => {
            const m = UNIVERSE_META[u.id] ?? {};
            const isActive = selected === u.id;
            return (
              <button
                key={u.id}
                onClick={() => u.available && setSelected(u.id)}
                style={{
                  ...styles.universeCard,
                  borderColor: isActive ? m.color : "#333",
                  backgroundColor: isActive ? `${m.color}18` : "#1a1a1a",
                  color: u.available ? "#eee" : "#555",
                  cursor: u.available ? "pointer" : "not-allowed",
                  transform: isActive ? "scale(1.03)" : "scale(1)",
                }}
                title={u.available ? `Switch to ${u.label}` : "Not yet trained"}
              >
                <span style={{ fontSize: 28 }}>{m.emoji}</span>
                <span style={{ ...styles.cardLabel, color: isActive ? m.color : "#aaa" }}>{u.label}</span>
                {!u.available && <span style={styles.notTrained}>not trained</span>}
                {u.available && (
                  <span style={{ ...styles.backendBadge, color: m.color }}>
                    {u.backend === "gpt2" ? "GPT-2" : "scratch"}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </section>

      {/* ── Chat history ── */}
      <main style={styles.chatArea}>
        {history.length === 0 && (
          <p style={styles.placeholder}>
            Select a universe, enter a prompt, and press Generate.
          </p>
        )}
        {history.map(entry => (
          <ChatEntry key={entry.id} entry={entry} meta={UNIVERSE_META[entry.universe] ?? {}} />
        ))}
        <div ref={bottomRef} />
      </main>

      {/* ── Error banner ── */}
      {error && (
        <div style={styles.errorBanner}>
          {error}
          <button style={styles.dismissBtn} onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {/* ── Input area ── */}
      <footer style={{ ...styles.inputArea, borderTopColor: meta.color + "44" }}>
        <div style={styles.settingsRow}>
          <label style={styles.settingLabel}>
            Max tokens
            <input
              type="number"
              min={32} max={512} step={32}
              value={maxTokens}
              onChange={e => setMaxTokens(Number(e.target.value))}
              style={styles.numberInput}
            />
          </label>
          <label style={styles.settingLabel}>
            Temperature
            <input
              type="number"
              min={0.1} max={2.0} step={0.05}
              value={temperature}
              onChange={e => setTemperature(Number(e.target.value))}
              style={styles.numberInput}
            />
          </label>
          <label style={{ ...styles.settingLabel, cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={useRag}
              onChange={e => setUseRag(e.target.checked)}
              style={{ accentColor: meta.color }}
            />
            Use RAG
          </label>
        </div>
        <div style={styles.promptRow}>
          <textarea
            style={styles.textarea}
            rows={3}
            placeholder={selected ? `Enter your ${UNIVERSE_META[selected]?.emoji ?? ""} story prompt…` : "Select a universe first"}
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={!selected || loading}
          />
          <button
            style={{
              ...styles.generateBtn,
              backgroundColor: meta.color,
              color: "#111",
              opacity: (!prompt.trim() || !selected || loading) ? 0.4 : 1,
            }}
            onClick={handleGenerate}
            disabled={!prompt.trim() || !selected || loading}
          >
            {loading ? "…" : "Generate"}
          </button>
        </div>
        <p style={styles.hint}>⌘ + Enter to generate</p>
      </footer>
    </div>
  );
}

function ChatEntry({ entry, meta }) {
  const [showPassages, setShowPassages] = useState(false);

  return (
    <div style={styles.chatEntry}>
      {/* User prompt bubble */}
      <div style={styles.promptBubble}>
        <span style={{ ...styles.universeBadge, color: meta.color }}>
          {meta.emoji} {entry.universe.replace("_", " ")}
        </span>
        <p style={styles.promptText}>{entry.prompt}</p>
      </div>

      {/* Model response */}
      {!entry.result ? (
        <div style={styles.generatingIndicator}>
          <span className="dot" />
          <span className="dot" style={{ animationDelay: "0.2s" }} />
          <span className="dot" style={{ animationDelay: "0.4s" }} />
        </div>
      ) : entry.result.error ? (
        <div style={{ ...styles.responseBubble, borderColor: "#ff4444" }}>
          <p style={{ color: "#ff4444" }}>Error: {entry.result.error}</p>
        </div>
      ) : (
        <div style={{ ...styles.responseBubble, borderColor: meta.color + "55" }}>
          <p style={styles.generatedText}>{entry.result.generated_text}</p>

          {/* Retrieved passages toggle */}
          {entry.result.retrieved_passages?.length > 0 && (
            <div style={styles.passagesSection}>
              <button
                style={{ ...styles.passagesToggle, color: meta.color }}
                onClick={() => setShowPassages(s => !s)}
              >
                {showPassages ? "▾" : "▸"} {entry.result.retrieved_passages.length} lore passages retrieved
              </button>
              {showPassages && (
                <div style={styles.passagesList}>
                  {entry.result.retrieved_passages.map((p, i) => (
                    <div key={i} style={{ ...styles.passageCard, borderLeftColor: meta.color }}>
                      <span style={{ ...styles.passageNum, color: meta.color }}>#{i + 1}</span>
                      <p style={styles.passageText}>{p}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Styles ─────────────────────────────────────────────────────────────────────

const styles = {
  root: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    background: "#0f0f0f",
    color: "#eee",
    fontFamily: "'Segoe UI', system-ui, sans-serif",
  },
  header: {
    padding: "20px 28px 12px",
    borderBottom: "1px solid #222",
    flexShrink: 0,
  },
  title: {
    margin: 0,
    fontSize: 26,
    fontWeight: 700,
    letterSpacing: "0.04em",
    color: "#fff",
  },
  subtitle: {
    margin: "4px 0 0",
    fontSize: 13,
    color: "#666",
  },
  pickerSection: {
    padding: "14px 28px",
    borderBottom: "1px solid #222",
    flexShrink: 0,
  },
  pickerRow: {
    display: "flex",
    gap: 12,
    flexWrap: "wrap",
  },
  universeCard: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 4,
    padding: "12px 20px",
    border: "2px solid",
    borderRadius: 10,
    background: "#1a1a1a",
    cursor: "pointer",
    transition: "all 0.15s ease",
    minWidth: 120,
  },
  cardLabel: {
    fontSize: 13,
    fontWeight: 600,
    letterSpacing: "0.03em",
  },
  notTrained: {
    fontSize: 10,
    color: "#555",
    letterSpacing: "0.05em",
  },
  backendBadge: {
    fontSize: 10,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    opacity: 0.8,
  },
  chatArea: {
    flex: 1,
    overflowY: "auto",
    padding: "20px 28px",
    display: "flex",
    flexDirection: "column",
    gap: 24,
  },
  placeholder: {
    color: "#444",
    textAlign: "center",
    marginTop: 60,
    fontSize: 15,
  },
  chatEntry: {
    display: "flex",
    flexDirection: "column",
    gap: 10,
  },
  promptBubble: {
    alignSelf: "flex-end",
    maxWidth: "70%",
    background: "#1e1e1e",
    border: "1px solid #333",
    borderRadius: "12px 12px 2px 12px",
    padding: "10px 14px",
  },
  universeBadge: {
    fontSize: 11,
    fontWeight: 700,
    textTransform: "uppercase",
    letterSpacing: "0.08em",
    display: "block",
    marginBottom: 4,
  },
  promptText: {
    margin: 0,
    fontSize: 14,
    lineHeight: 1.5,
    color: "#ddd",
  },
  generatingIndicator: {
    alignSelf: "flex-start",
    display: "flex",
    gap: 6,
    padding: "12px 16px",
    background: "#1a1a1a",
    border: "1px solid #2a2a2a",
    borderRadius: "2px 12px 12px 12px",
  },
  responseBubble: {
    alignSelf: "flex-start",
    maxWidth: "85%",
    background: "#141414",
    border: "1px solid",
    borderRadius: "2px 12px 12px 12px",
    padding: "14px 18px",
  },
  generatedText: {
    margin: 0,
    fontSize: 15,
    lineHeight: 1.75,
    color: "#e0e0e0",
    whiteSpace: "pre-wrap",
  },
  passagesSection: {
    marginTop: 12,
    borderTop: "1px solid #2a2a2a",
    paddingTop: 10,
  },
  passagesToggle: {
    background: "none",
    border: "none",
    cursor: "pointer",
    fontSize: 12,
    fontWeight: 600,
    letterSpacing: "0.04em",
    padding: 0,
  },
  passagesList: {
    display: "flex",
    flexDirection: "column",
    gap: 8,
    marginTop: 8,
  },
  passageCard: {
    borderLeft: "3px solid",
    paddingLeft: 10,
    paddingTop: 4,
    paddingBottom: 4,
  },
  passageNum: {
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: "0.08em",
    display: "block",
    marginBottom: 2,
  },
  passageText: {
    margin: 0,
    fontSize: 12,
    color: "#999",
    lineHeight: 1.6,
  },
  errorBanner: {
    background: "#2a0a0a",
    border: "1px solid #ff4444",
    color: "#ff8888",
    padding: "10px 16px",
    margin: "0 28px",
    borderRadius: 8,
    fontSize: 13,
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    flexShrink: 0,
  },
  dismissBtn: {
    background: "none",
    border: "none",
    color: "#ff8888",
    cursor: "pointer",
    fontSize: 16,
    padding: "0 4px",
  },
  inputArea: {
    padding: "14px 28px 20px",
    borderTop: "1px solid",
    background: "#0f0f0f",
    flexShrink: 0,
  },
  settingsRow: {
    display: "flex",
    gap: 20,
    marginBottom: 10,
  },
  settingLabel: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    fontSize: 12,
    color: "#666",
  },
  numberInput: {
    width: 70,
    background: "#1a1a1a",
    border: "1px solid #333",
    borderRadius: 6,
    color: "#ccc",
    padding: "4px 8px",
    fontSize: 12,
  },
  promptRow: {
    display: "flex",
    gap: 10,
    alignItems: "flex-end",
  },
  textarea: {
    flex: 1,
    background: "#1a1a1a",
    border: "1px solid #333",
    borderRadius: 10,
    color: "#eee",
    padding: "10px 14px",
    fontSize: 14,
    resize: "none",
    lineHeight: 1.5,
    fontFamily: "inherit",
    outline: "none",
  },
  generateBtn: {
    padding: "0 22px",
    height: 44,
    border: "none",
    borderRadius: 10,
    fontWeight: 700,
    fontSize: 14,
    cursor: "pointer",
    letterSpacing: "0.04em",
    flexShrink: 0,
    transition: "opacity 0.15s",
  },
  hint: {
    margin: "6px 0 0",
    fontSize: 11,
    color: "#444",
  },
};
