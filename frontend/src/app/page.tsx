
"use client";

import { useState } from "react";

export default function Home() {
  const [gameId, setGameId] = useState("");
  const [data, setData] = useState<any[] | null>(null);

  const fetchData = async () => {
    if (!gameId) return;
    try {
      const res = await fetch(`http://127.0.0.1:8000/game/${gameId}`);
      const json = await res.json();
      setData(json);
    } catch (err) {
      console.error(err);
    }
  };

  return (
      <div
          style={{
            minHeight: "100vh",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            backgroundColor: "#1a1c3a",
            color: "white",
            fontFamily: "Arial, sans-serif",
          }}
      >
        <h1 style={{ fontSize: "3rem", fontWeight: "bold", marginBottom: "0.5rem" }}>
          moonshot
        </h1>
        <p style={{ fontStyle: "italic", color: "#ccc", marginBottom: "1.5rem" }}>
          Enter Game ID
        </p>

        <div
            style={{
              display: "flex",
              alignItems: "center",
              backgroundColor: "#2a2d55",
              borderRadius: "50px",
              padding: "0.5rem 1rem",
              width: "320px",
            }}
        >
          <input
              type="number"
              placeholder="Game ID..."
              value={gameId}
              onChange={(e) => setGameId(e.target.value)}
              style={{
                flex: 1,
                background: "transparent",
                border: "none",
                outline: "none",
                color: "white",
                fontSize: "1rem",
                padding: "0.5rem",
              }}
          />
          <button
              onClick={fetchData}
              style={{
                backgroundColor: "#4a6cf7",
                border: "none",
                color: "white",
                fontSize: "1.2rem",
                borderRadius: "50%",
                padding: "0.5rem 0.7rem",
                cursor: "pointer",
                transition: "background 0.3s ease",
              }}
              onMouseOver={(e) => (e.currentTarget.style.backgroundColor = "#3651d4")}
              onMouseOut={(e) => (e.currentTarget.style.backgroundColor = "#4a6cf7")}
          >
            &#10140;
          </button>
        </div>

          {data && (
              <div
                  style={{
                      marginTop: "2rem",
                      width: "80%",
                      maxWidth: "600px",
                      backgroundColor: "#2a2d55",
                      padding: "1.5rem",
                      borderRadius: "12px",
                      color: "white",
                      fontSize: "1rem",
                      textAlign: "left",
                  }}
              >
                  <h2 style={{ marginBottom: "1rem", color: "#4a90e2" }}>
                      Pitch Details
                  </h2>
                  {data.slice(0, 5).map((row: any, i: number) => (
                      <div
                          key={i}
                          style={{
                              marginBottom: "1rem",
                              paddingBottom: "0.8rem",
                              borderBottom: "1px solid #444",
                          }}
                      >
                          <p><strong>Batter:</strong> {row.player_name}</p>
                          <p><strong>Pitch Type:</strong> {row.pitch_type}</p>
                          <p><strong>Speed:</strong> {row.release_speed} mph</p>
                          <p><strong>Description:</strong> {row.des}</p>
                          <p><strong>Zone:</strong> {row.zone}</p>
                      </div>
                  ))}
              </div>
          )}
      </div>
  );
}
