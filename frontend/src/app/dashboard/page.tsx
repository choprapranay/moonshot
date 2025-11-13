"use client";

import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { StrikeZoneHeatmap } from "@/components/StrikeZoneHeatmap";

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

type HeatmapPoint = {
  plate_x: number;
  plate_z: number;
  expected_value_diff: number;
};

type SwingPoint = {
  plate_x: number;
  plate_z: number;
  pitch_type?: string | null;
  description?: string | null;
};

type PlayerHeatmapResponse = {
  heatmap: HeatmapPoint[];
  swings: SwingPoint[];
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
  const [heatmapData, setHeatmapData] = useState<HeatmapPoint[]>([]);
  const [swingData, setSwingData] = useState<SwingPoint[]>([]);
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [heatmapError, setHeatmapError] = useState<string | null>(null);

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

  useEffect(() => {
    if (!gameIdParam || !selectedPlayerId) {
      setHeatmapData([]);
      setSwingData([]);
      return;
    }

    const abort = new AbortController();
    setHeatmapLoading(true);
    setHeatmapError(null);

    fetch(`${API_BASE_URL}/game/${gameIdParam}/player/${selectedPlayerId}/heatmap`, {
      signal: abort.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(await response.text());
        }
        return response.json() as Promise<PlayerHeatmapResponse>;
      })
      .then((payload) => {
        setHeatmapData(payload.heatmap ?? []);
        setSwingData(payload.swings ?? []);
      })
      .catch((err: unknown) => {
        if (abort.signal.aborted) return;
        const message = err instanceof Error ? err.message : "Unable to load heatmap";
        setHeatmapError(message);
        setHeatmapData([]);
        setSwingData([]);
      })
      .finally(() => {
        if (!abort.signal.aborted) {
          setHeatmapLoading(false);
        }
      });

    return () => {
      abort.abort();
    };
  }, [gameIdParam, selectedPlayerId]);

  return (
    <div
      className="relative min-h-screen overflow-hidden text-white"
      style={{ backgroundColor: "#07090f" }}
    >
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.44]"
        style={{
          backgroundImage:
            "radial-gradient(at 24% 18%, hsla(212, 95%, 32%, 0.35) 0px, transparent 55%), radial-gradient(at 78% 18%, hsla(340, 90%, 37%, 0.32) 0px, transparent 55%), radial-gradient(at 50% 82%, hsla(240, 82%, 38%, 0.34) 0px, transparent 60%)",
        }}
      />
      <div className="absolute inset-0 pointer-events-none backdrop-blur-[28px] opacity-85" />

      <div
        className="relative z-10 mx-auto flex h-full max-w-7xl flex-col gap-6 px-8 py-6 md:px-12 md:py-10"
        style={{ paddingTop: "10vh", paddingBottom: "10vh" }}
      >
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
               style={{
                 color: "#d6daf6",
                 background: "rgba(7, 10, 22, 0.52)",
                 boxShadow: "0 14px 32px rgba(3, 5, 12, 0.6)",
                 backdropFilter: "blur(26px) saturate(140%)",
                 WebkitBackdropFilter: "blur(26px) saturate(140%)",
               }}
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
            {/* <button
              type="button"
              className="flex items-center gap-2 rounded-full px-4 py-2 text-sm"
              style={{ backgroundColor: "#2a2d55", color: "#d6daf6" }}
            >
              <span className="material-symbols-outlined text-base">calendar_today</span>
              <span>Last 30 Days</span>
              <span className="material-symbols-outlined text-base">expand_more</span>
            </button> */}
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
            className="relative flex flex-1 items-center justify-center overflow-hidden rounded-2xl"
            style={{
              background: "rgba(3, 5, 15, 0.56)",
              boxShadow: "0 26px 70px rgba(4, 6, 14, 0.7)",
              backdropFilter: "blur(34px) saturate(135%)",
              WebkitBackdropFilter: "blur(34px) saturate(135%)",
            }}
          >
            <div
              className="pointer-events-none absolute inset-0 rounded-2xl"
              style={{
                background:
                  "linear-gradient(135deg, rgba(102, 186, 255, 0.28) 0%, rgba(236, 132, 205, 0.18) 45%, rgba(98, 158, 255, 0.22) 100%)",
                opacity: 0.28,
              }}
            />
            <div className="relative z-10">
              <p style={{ color: "#d6daf6" }}>Loading dashboard…</p>
            </div>
          </div>
        ) : (
          <div className="flex flex-1 flex-col gap-6 lg:flex-row">
            <section className="flex flex-1 flex-col gap-6">
              <div
                className="flex items-center gap-4 overflow-x-auto overflow-y-visible py-2"
                style={{ paddingLeft: "12px", paddingRight: "12px" }}
              >
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
                        padding: "4px",
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
                className="relative flex flex-1 items-center justify-center overflow-hidden rounded-2xl"
                style={{
                  background: "rgba(3, 5, 15, 0.58)",
                  boxShadow: "0 28px 70px rgba(3, 5, 12, 0.78)",
                  backdropFilter: "blur(40px) saturate(135%)",
                  WebkitBackdropFilter: "blur(40px) saturate(135%)",
                  minHeight: "460px",
                }}
              >
                <div
                  className="pointer-events-none absolute inset-0 rounded-2xl"
                  style={{
                    background:
                      "linear-gradient(140deg, rgba(102, 186, 255, 0.28) 0%, rgba(236, 132, 205, 0.18) 42%, rgba(98, 158, 255, 0.24) 100%)",
                    opacity: 0.32,
                    mixBlendMode: "screen",
                  }}
                />
                <div className="relative z-10 flex h-full w-full items-center justify-center">
                  {heatmapLoading ? (
                    <span className="text-sm" style={{ color: "#9aa0d4" }}>
                      Loading strike zone…
                    </span>
                  ) : heatmapError ? (
                    <span className="text-sm text-red-300">{heatmapError}</span>
                  ) : heatmapData.length ? (
                    <StrikeZoneHeatmap heatmap={heatmapData} swings={swingData} />
                  ) : (
                    <span className="text-sm" style={{ color: "#9aa0d4" }}>
                      No heatmap data available.
                    </span>
                  )}
                </div>
              </div>
            </section>

            <aside
              className="relative flex w-full flex-col gap-6 overflow-hidden rounded-2xl p-6 lg:w-[360px]"
              style={{
                background: "rgba(4, 6, 18, 0.56)",
                boxShadow: "0 32px 82px rgba(3, 5, 12, 0.78)",
                backdropFilter: "blur(44px) saturate(140%)",
                WebkitBackdropFilter: "blur(44px) saturate(140%)",
                minHeight: "460px",
              }}
            >
              <div
                className="pointer-events-none absolute inset-0 rounded-2xl"
                style={{
                  background:
                    "linear-gradient(145deg, rgba(102, 186, 255, 0.28) 0%, rgba(236, 132, 205, 0.18) 45%, rgba(98, 158, 255, 0.22) 100%)",
                  opacity: 0.28,
                  mixBlendMode: "screen",
                }}
              />
              <div className="relative z-10 flex h-full flex-col gap-6">
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
                     style={{
                       background: "rgba(4, 7, 18, 0.5)",
                       boxShadow: "0 18px 44px rgba(2, 4, 10, 0.64)",
                       backdropFilter: "blur(36px) saturate(145%)",
                       WebkitBackdropFilter: "blur(36px) saturate(145%)",
                     }}
                   >
                    <h3 className="text-base font-semibold">Area of Concern</h3>
                    <div className="mt-3 flex items-baseline justify-between gap-4">
                      <span className="text-3xl font-bold text-slate-400">
                        —
                      </span>
                    </div>
                  </div>
                </>
              ) : (
                <div className="flex flex-1 items-center justify-center text-sm" style={{ color: "#9aa0d4" }}>
                  Select a player to see detailed metrics.
                </div>
              )}
              </div>
            </aside>
          </div>
        )}
      </div>
    </div>
  );
}

