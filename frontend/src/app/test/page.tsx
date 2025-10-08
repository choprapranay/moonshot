"use client";

import { useState } from "react";

type PitchData = {
  pitch_type: string;
  speed: number | null;
  description: string | null;
  player_id: string;
  diagram_index: number | null;
};

type GameData = {
  game_date: string;
  home_team: string;
  away_team: string;
};

type FetchGameDataAndSaveResponse = {
  game_id: string;
  gameData: GameData;
  pitches: PitchData[];
};

export default function TestPage() {
  const [gameId, setGameId] = useState("");
  const [data, setData] = useState<FetchGameDataAndSaveResponse | null>(null);
  const [selectedDiagramIndex, setSelectedDiagramIndex] = useState<number | null>(null);
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
      const json: FetchGameDataAndSaveResponse = await response.json();
      setData(json);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const gameDescription = data
    ? `${data.gameData.game_date} — ${data.gameData.home_team} vs ${data.gameData.away_team}`
    : "";

  return (
    <div style={{ margin: "0 auto", color: "#fff", background: "#0e1224", minHeight: "100vh" }}>
      <div style={{ padding: "5vh 10vw", maxHeight: "100vh", display: "flex", flexDirection: "column", gap: 16 }}>
        <h1 style={{ flex: "0 0 auto", fontSize: 28 }}>Game Data Test</h1>
        
        <div style={{ flex: "0 0 auto", display: "flex", gap: 4 }}>
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
          <div style={{ flex: 1, padding: 12, background: "#ff4444", borderRadius: 6, marginBottom: 16 }}>
            Error: {error}
          </div>
        )}

        {data && (
          <div className="flex-1 p-4 rounded-lg border border-[#2a2d55] bg-[#1a1e3a]">
            <div className="flex gap-4 h-[70vh]">
              <div className="flex-1 flex flex-col gap-3">
                <div className="text-base">
                  <strong>Game:</strong> {gameDescription}
                </div>
                <div className="flex-1 overflow-y-auto">
                  <ul className="list-disc pl-5 m-0">
                    {data.pitches.map((p, idx) => (
                      <li
                        key={idx}
                        className="mb-2 cursor-pointer hover:underline"
                        onClick={() => setSelectedDiagramIndex(p.diagram_index ?? null)}
                      >
                        <span className="font-semibold">{p.pitch_type}</span>
                        <span> — {p.speed ?? "N/A"} mph</span>
                        {p.description ? <span> — {p.description}</span> : null}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              <div className="w-1/3 min-w-[280px]">
                <div className="bg-[#14183a] border border-[#2a2d55] rounded-lg p-3">
                  <div className="font-semibold mb-2">Pitch Type Distribution</div>
                  <div className="h-[200px] bg-[#0f1330] rounded mb-3" />
                  <div className="text-sm text-[#aab]">
                    <span className="font-semibold">Diagram Index:</span> {selectedDiagramIndex ?? "—"}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

