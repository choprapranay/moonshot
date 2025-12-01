import { GameAnalysisResponse } from "../entities/Game";
import { PlayerHeatmapResponse } from "../entities/Heatmap";

export interface IBaseballRepository {
  getGameAnalysis(gameId: string): Promise<GameAnalysisResponse>;
  getPlayerHeatmap(gameId: string, playerId: number): Promise<PlayerHeatmapResponse>;
}

