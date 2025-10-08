"use client";

import { useState } from "react";

export default function TestPage() {
  const [gameId, setGameId] = useState("");
  const [gameData, setGameData] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchGameData = async () => {
    if (!gameId) return;

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`http://127.0.0.1:8000/game-data/${gameId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setGameData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
      setGameData(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: "0 auto", color: "#fff", background: "#0e1224", minHeight: "100vh" }}>
      <h1 style={{ fontSize: 28, marginBottom: 12 }}>Game Data Test</h1>
      
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input
          value={gameId}
          onChange={(e) => setGameId(e.target.value)}
          placeholder="Enter Game ID"
          style={{ flex: 1, padding: 8, borderRadius: 6, border: "1px solid #2a2d55", background: "#171a36", color: "#fff" }}
        />
        <button 
          onClick={fetchGameData} 
          disabled={loading || !gameId}
          style={{ 
            padding: "8px 12px", 
            borderRadius: 6, 
            border: 0, 
            background: loading ? "#666" : "#4a6cf7", 
            color: "#fff", 
            cursor: loading ? "not-allowed" : "pointer" 
          }}
        >
          {loading ? "Loading..." : "Fetch Game Data"}
        </button>
      </div>

      {error && (
        <div style={{ padding: 12, background: "#ff4444", borderRadius: 6, marginBottom: 16 }}>
          Error: {error}
        </div>
      )}

      {gameData && (
        <div style={{ padding: 16, background: "#1a1e3a", border: "1px solid #2a2d55", borderRadius: 8, maxWidth: 400 }}>
          <div><strong>ID:</strong> {gameData.id}</div>
          <div><strong>Game Date:</strong> {gameData.game_date}</div>
          <div><strong>Home Team:</strong> {gameData.home_team}</div>
          <div><strong>Away Team:</strong> {gameData.away_team}</div>
          <div><strong>Game PK:</strong> {gameData.game_pk}</div>
        </div>
      )}
    </div>
  );
}
