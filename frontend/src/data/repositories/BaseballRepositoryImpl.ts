import { IBaseballRepository } from "@/domain/repositories/IBaseballRepository";
import { GameAnalysisResponse } from "@/domain/entities/Game";
import { PlayerHeatmapResponse } from "@/domain/entities/Heatmap";

export class BaseballRepositoryImpl implements IBaseballRepository {
  private readonly baseUrl: string;

  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl ?? process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
  }

  async getGameAnalysis(gameId: string): Promise<GameAnalysisResponse> {
    const response = await fetch(`${this.baseUrl}/game/${gameId}/analysis`);
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || "Unable to fetch game analysis");
    }
    return response.json();
  }

  async getPlayerHeatmap(gameId: string, playerId: number): Promise<PlayerHeatmapResponse> {
    const response = await fetch(`${this.baseUrl}/game/${gameId}/player/${playerId}/heatmap`);
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || "Unable to fetch player heatmap");
    }
    return response.json();
  }
}

