"use client";

import { useEffect, useState } from "react";

type Game = {
  id: string;
  game_date: string;
  home_team: string | null;
  away_team: string | null;
};

export default function DataPage() {
  const [name, setName] = useState("");
  const [items, setItems] = useState<Game[]>([]);
  const [loading, setLoading] = useState(false);

  const loadItems = async () => {
    const res = await fetch("http://127.0.0.1:8000/retrieveFromTmpData");
    const data = await res.json();
    setItems(Array.isArray(data) ? data : []);
  };

  const addItem = async () => {
    if (!name) return;
    setLoading(true);
    await fetch("http://127.0.0.1:8000/addToTmpData", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name })
    });
    setName("");
    await loadItems();
    setLoading(false);
  };

  useEffect(() => {
    loadItems();
  }, []);

  return (
    <div style={{ padding: 24, maxWidth: 640, margin: "0 auto", color: "#fff", background: "#0e1224", minHeight: "100vh" }}>
      <h1 style={{ fontSize: 28, marginBottom: 12 }}>Tmp Data</h1>
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Item name"
          style={{ flex: 1, padding: 8, borderRadius: 6, border: "1px solid #2a2d55", background: "#171a36", color: "#fff" }}
        />
        <button onClick={addItem} disabled={loading} style={{ padding: "8px 12px", borderRadius: 6, border: 0, background: "#4a6cf7", color: "#fff", cursor: "pointer" }}>
          {loading ? "Adding..." : "Add"}
        </button>
      </div>
      <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: 8 }}>
        {items.map((it) => (
          <li key={it.id} style={{ padding: 12, background: "#1a1e3a", border: "1px solid #2a2d55", borderRadius: 8 }}>
            <div style={{ fontWeight: 600 }}>{it.home_team ?? "—"} vs {it.away_team ?? "—"}</div>
            <div style={{ fontSize: 12, color: "#b9bed6" }}>{new Date(it.game_date).toLocaleDateString()}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}


