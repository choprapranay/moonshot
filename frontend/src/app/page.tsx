
"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();
  const [gameId, setGameId] = useState("");

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!gameId) return;
    router.push(`/dashboard?gameId=${gameId}`);
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
          }}
      >
        <h1 style={{ fontSize: "3rem", fontWeight: "bold", marginBottom: "0.5rem", color: "white" }}>
          moonshot
        </h1>
        <p style={{ fontStyle: "italic", color: "#ccc", marginBottom: "1.5rem" }}>
          Enter Game ID
        </p>
        <form
            onSubmit={handleSubmit}
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
              type="submit"
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
        </form>
      </div>
  );
}
