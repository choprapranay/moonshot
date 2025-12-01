import { IBaseballRepository } from "../repositories/IBaseballRepository";
import { GameAnalysisResponse } from "../entities/Game";

export class GetGameAnalysisUseCase {
  constructor(private repository: IBaseballRepository) {}

  async execute(gameId: string): Promise<GameAnalysisResponse> {
    return this.repository.getGameAnalysis(gameId);
  }
}

