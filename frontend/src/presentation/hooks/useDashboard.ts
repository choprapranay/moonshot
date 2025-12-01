import { useState, useEffect, useMemo } from "react";
import { BaseballRepositoryImpl } from "@/data/repositories/BaseballRepositoryImpl";
import { GetGameAnalysisUseCase } from "@/domain/usecases/GetGameAnalysisUseCase";
import { GetPlayerHeatmapUseCase } from "@/domain/usecases/GetPlayerHeatmapUseCase";
import { GameAnalysisResponse } from "@/domain/entities/Game";
import { PlayerSummary } from "@/domain/entities/Player";
import { HeatmapPoint, SwingPoint } from "@/domain/entities/Heatmap";

// We could inject these, but for now we instantiate them here
const repository = new BaseballRepositoryImpl();
const getGameAnalysisUseCase = new GetGameAnalysisUseCase(repository);
const getPlayerHeatmapUseCase = new GetPlayerHeatmapUseCase(repository);

export function useDashboard(gameIdParam: string | null) {
  const [analysis, setAnalysis] = useState<GameAnalysisResponse | null>(null);
  const [selectedTeam, setSelectedTeam] = useState<string | null>(null);
  const [selectedPlayerId, setSelectedPlayerId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [heatmapData, setHeatmapData] = useState<HeatmapPoint[]>([]);
  const [swingData, setSwingData] = useState<SwingPoint[]>([]);
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [heatmapError, setHeatmapError] = useState<string | null>(null);
  const [selectedSwingIndex, setSelectedSwingIndex] = useState<number | null>(null);

  // Fetch Game Analysis
  useEffect(() => {
    if (!gameIdParam) return;

    setLoading(true);
    setError(null);

    getGameAnalysisUseCase.execute(gameIdParam)
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

  // Filter and Sort Players
  const playersForTeam = useMemo(() => {
    if (!analysis) {
      return [] as PlayerSummary[];
    }
    if (!selectedTeam) {
      return analysis.players;
    }
    const filteredPlayers = analysis.players.filter((player) => player.team === selectedTeam);

    // Sort specific players to the front (Presentation Logic)
    filteredPlayers.sort((a, b) => {
      // Teoscar Hernandez
      if (a.player_id === 606192) return -1;
      if (b.player_id === 606192) return 1;
      return 0;
    });

    return filteredPlayers;
  }, [analysis, selectedTeam]);

  // Select default player when team changes
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

  // Fetch Heatmap
  useEffect(() => {
    if (!gameIdParam || !selectedPlayerId) {
      setHeatmapData([]);
      setSwingData([]);
      return;
    }

    // Note: AbortController logic is tricky with Promise-based UseCases unless they support cancellation.
    // We'll implement a simple version here, assuming UseCase is fast or we ignore stale results.
    // Ideally, pass AbortSignal to Repository -> UseCase.
    // For now, we'll keep the "ignore stale" pattern using a flag.
    
    let active = true;
    setHeatmapLoading(true);
    setHeatmapError(null);

    getPlayerHeatmapUseCase.execute(gameIdParam, selectedPlayerId)
      .then((payload) => {
        if (!active) return;
        setHeatmapData(payload.heatmap ?? []);
        setSwingData(payload.swings ?? []);
        setSelectedSwingIndex(null);
      })
      .catch((err: unknown) => {
        if (!active) return;
        const message = err instanceof Error ? err.message : "Unable to load heatmap";
        setHeatmapError(message);
        setHeatmapData([]);
        setSwingData([]);
      })
      .finally(() => {
        if (active) {
          setHeatmapLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [gameIdParam, selectedPlayerId]);

  return {
    analysis,
    selectedTeam,
    setSelectedTeam,
    selectedPlayerId,
    setSelectedPlayerId,
    loading,
    error,
    heatmapData,
    swingData,
    heatmapLoading,
    heatmapError,
    selectedSwingIndex,
    setSelectedSwingIndex,
    playersForTeam,
    selectedPlayer
  };
}

