"use client";

import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";

type TeamInfo = {
  code: string;
  name: string;
};

type PlayerStats = {
  pitches_seen: number;
  swing_percentage: number;
  take_percentage: number;
  whiff_percentage: number;
  contact_percentage: number;
  average_velocity: number | null;
};

type PlayerSummary = {
  player_id: number;
  player_name: string;
  team: string;
  headshot_url?: string | null;
  stats: PlayerStats;
  impact_zone_delta?: number | null;
};

type GameAnalysisResponse = {
  game_id: number;
  game_date: string;
  teams: TeamInfo[];
  players: PlayerSummary[];
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
const FALLBACK_HEADSHOT = "https://via.placeholder.com/80x80?text=?";

async function fetchGameAnalysis(gameId: string): Promise<GameAnalysisResponse> {
  const response = await fetch(`${API_BASE_URL}/game/${gameId}/analysis`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Unable to fetch game analysis");
  }
  return response.json();
}

export default function Dashboard() {
  const searchParams = useSearchParams();
  const gameIdParam = searchParams.get("gameId");

  const [analysis, setAnalysis] = useState<GameAnalysisResponse | null>(null);
  const [selectedTeam, setSelectedTeam] = useState<string | null>(null);
  const [selectedPlayerId, setSelectedPlayerId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!gameIdParam) {
      return;
    }

    setLoading(true);
    setError(null);

    fetchGameAnalysis(gameIdParam)
      .then((payload) => {
        setAnalysis(payload);
        const defaultTeam = payload.teams[0]?.code ?? null;
        setSelectedTeam(defaultTeam);
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : "Unknown error";
        setError(message);
        setAnalysis(null);
      })
      .finally(() => setLoading(false));
  }, [gameIdParam]);

  const playersForTeam = useMemo(() => {
    if (!analysis) {
      return [] as PlayerSummary[];
    }
    if (!selectedTeam) {
      return analysis.players;
    }
    return analysis.players.filter((player) => player.team === selectedTeam);
  }, [analysis, selectedTeam]);

  useEffect(() => {
    if (!playersForTeam.length) {
      setSelectedPlayerId(null);
      return;
    }

    if (!selectedPlayerId || !playersForTeam.some((player) => player.player_id === selectedPlayerId)) {
      setSelectedPlayerId(playersForTeam[0].player_id);
    }
  }, [playersForTeam, selectedPlayerId]);

  const selectedPlayer = useMemo(() => {
    if (!playersForTeam.length || selectedPlayerId === null) {
      return null;
    }
    return playersForTeam.find((player) => player.player_id === selectedPlayerId) ?? playersForTeam[0];
  }, [playersForTeam, selectedPlayerId]);

  return (
    <div
      className="min-h-screen text-white"
      style={{ backgroundColor: "#1a1c3a" }}
    >
      <div className="mx-auto flex h-full max-w-6xl flex-col gap-6 p-6 md:p-10">
        <header className="flex flex-col justify-between gap-4 md:flex-row md:items-center">
          <div>
            <h1
              className="text-3xl font-semibold md:text-4xl"
              style={{ color: "#8bd3ff" }}
            >
              moonshot
            </h1>
            <p className="mt-1 text-sm" style={{ color: "#b0b4d1" }}>
              {analysis ? `Game ${analysis.game_id} • ${analysis.game_date}` : "Awaiting game data"}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <label
              className="flex items-center gap-2 rounded-full px-4 py-2 text-sm"
              style={{ backgroundColor: "#2a2d55", color: "#d6daf6" }}
            >
              <span className="hidden text-xs uppercase tracking-wide md:inline" style={{ color: "#8d92c2" }}>
                Team
              </span>
              <select
                className="bg-transparent text-sm font-medium outline-none"
                style={{ color: "#ffffff" }}
                value={selectedTeam ?? ""}
                onChange={(event) => setSelectedTeam(event.target.value || null)}
                disabled={!analysis?.teams.length}
              >
                {analysis?.teams.map((team) => (
                  <option className="text-black" key={team.code} value={team.code}>
                    {team.name}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className="flex items-center gap-2 rounded-full px-4 py-2 text-sm"
              style={{ backgroundColor: "#2a2d55", color: "#d6daf6" }}
            >
              <span className="material-symbols-outlined text-base">calendar_today</span>
              <span>Last 30 Days</span>
              <span className="material-symbols-outlined text-base">expand_more</span>
            </button>
          </div>
        </header>

        {error && (
          <div
            className="rounded-xl px-4 py-3 text-sm"
            style={{
              border: "1px solid #ff6b6b",
              background: "rgba(255, 107, 107, 0.15)",
              color: "#ffdcdc",
            }}
          >
            {error}
          </div>
        )}

        {loading ? (
          <div
            className="flex flex-1 items-center justify-center rounded-2xl border border-dashed"
            style={{ borderColor: "#3a3e6b", backgroundColor: "#2a2d55" }}
          >
            <p style={{ color: "#d6daf6" }}>Loading dashboard…</p>
          </div>
        ) : (
          <div className="flex flex-1 flex-col gap-6 lg:flex-row">
            <section className="flex flex-1 flex-col gap-6">
              <div className="flex items-center gap-4 overflow-x-auto pb-2">
                <div className="flex flex-nowrap items-center gap-4">
                  {playersForTeam.map((player) => (
                    <button
                      key={player.player_id}
                      type="button"
                      onClick={() => setSelectedPlayerId(player.player_id)}
                      className={`relative flex-shrink-0 rounded-full border-2 transition-all ${
                        player.player_id === selectedPlayerId
                          ? "ring-4"
                          : "hover:border-[#4a6cf7]"
                      }`}
                      style={{
                        borderColor:
                          player.player_id === selectedPlayerId ? "#4a6cf7" : "transparent",
                        boxShadow:
                          player.player_id === selectedPlayerId
                            ? "0 0 0 6px rgba(74, 108, 247, 0.25)"
                            : undefined,
                      }}
                    >
                      <img
                        src={player.headshot_url ?? FALLBACK_HEADSHOT}
                        alt={player.player_name}
                        className="h-14 w-14 rounded-full object-cover"
                      />
                    </button>
                  ))}
                  {!playersForTeam.length && (
                    <span className="text-sm" style={{ color: "#9aa0d4" }}>
                      No batters available for this team.
                    </span>
                  )}
                </div>
              </div>

              <div
                className="flex flex-1 items-center justify-center rounded-2xl border border-dashed"
                style={{ borderColor: "#3a3e6b", backgroundColor: "#2a2d55" }}
              >
                <span className="text-sm" style={{ color: "#9aa0d4" }}>
                  Heatmap coming soon
                </span>
              </div>
            </section>

            <aside
              className="flex w-full flex-col gap-6 rounded-2xl p-6 lg:w-[360px]"
              style={{ backgroundColor: "#2a2d55" }}
            >
              {selectedPlayer ? (
                <>
                  <div className="flex items-center gap-4">
                    <img
                      src={selectedPlayer.headshot_url ?? FALLBACK_HEADSHOT}
                      alt={selectedPlayer.player_name}
                      className="h-20 w-20 rounded-full object-cover"
                    />
                    <div>
                      <h2 className="text-2xl font-semibold text-white">{selectedPlayer.player_name}</h2>
                      <p className="text-xs uppercase tracking-wide text-slate-400">
                        #{selectedPlayer.player_id} | {selectedPlayer.team}
                      </p>
                    </div>
                  </div>

                  <dl className="grid grid-cols-2 gap-x-6 gap-y-4 text-white">
                    <div>
                      <dt className="text-xs uppercase tracking-wide" style={{ color: "#9aa0d4" }}>
                        Pitches Seen
                      </dt>
                      <dd className="text-xl font-semibold">{selectedPlayer.stats.pitches_seen}</dd>
                    </div>
                    <div>
                      <dt className="text-xs uppercase tracking-wide" style={{ color: "#9aa0d4" }}>
                        Avg. Velo
                      </dt>
                      <dd className="text-xl font-semibold">
                        {selectedPlayer.stats.average_velocity !== null
                          ? `${selectedPlayer.stats.average_velocity.toFixed(1)} mph`
                          : "—"}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-xs uppercase tracking-wide" style={{ color: "#9aa0d4" }}>
                        Swing %
                      </dt>
                      <dd className="text-xl font-semibold">{selectedPlayer.stats.swing_percentage.toFixed(1)}%</dd>
                    </div>
                    <div>
                      <dt className="text-xs uppercase tracking-wide" style={{ color: "#9aa0d4" }}>
                        Whiff %
                      </dt>
                      <dd className="text-xl font-semibold">{selectedPlayer.stats.whiff_percentage.toFixed(1)}%</dd>
                    </div>
                    <div>
                      <dt className="text-xs uppercase tracking-wide" style={{ color: "#9aa0d4" }}>
                        Take %
                      </dt>
                      <dd className="text-xl font-semibold">{selectedPlayer.stats.take_percentage.toFixed(1)}%</dd>
                    </div>
                    <div>
                      <dt className="text-xs uppercase tracking-wide" style={{ color: "#9aa0d4" }}>
                        Contact %
                      </dt>
                      <dd className="text-xl font-semibold">{selectedPlayer.stats.contact_percentage.toFixed(1)}%</dd>
                    </div>
                  </dl>

                  <div
                    className="mt-auto rounded-xl p-4"
                    style={{ backgroundColor: "#1a1c3a" }}
                  >
                    <h3 className="text-base font-semibold">Impact Zone Analysis</h3>
                    <div className="mt-3 flex items-baseline justify-between gap-4">
                      <span
                        className={`text-3xl font-bold ${
                          selectedPlayer.impact_zone_delta !== null && selectedPlayer.impact_zone_delta !== undefined
                            ? selectedPlayer.impact_zone_delta < 0
                              ? "text-red-400"
                              : "text-emerald-400"
                            : "text-slate-400"
                        }`}
                      >
                        {selectedPlayer.impact_zone_delta !== null && selectedPlayer.impact_zone_delta !== undefined
                          ? selectedPlayer.impact_zone_delta.toFixed(1)
                          : "—"}
                      </span>
                      <p className="text-sm" style={{ color: "#9aa0d4" }}>
                        Runs added (negative indicates runs lost) on swings high &amp; inside over the selected window.
                      </p>
                    </div>
                  </div>
                </>
              ) : (
                <div className="flex flex-1 items-center justify-center text-sm" style={{ color: "#9aa0d4" }}>
                  Select a player to see detailed metrics.
                </div>
              )}
            </aside>
          </div>
        )}
      </div>
    </div>
  );
}

