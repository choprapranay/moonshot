import { IBaseballRepository } from "../repositories/IBaseballRepository";
import { PlayerHeatmapResponse } from "../entities/Heatmap";

export class GetPlayerHeatmapUseCase {
  constructor(private repository: IBaseballRepository) {}

  async execute(gameId: string, playerId: number): Promise<PlayerHeatmapResponse> {
    return this.repository.getPlayerHeatmap(gameId, playerId);
  }
}

