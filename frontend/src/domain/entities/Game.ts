import { PlayerSummary } from "./Player";

export type TeamInfo = {
  code: string;
  name: string;
};

export type GameAnalysisResponse = {
  game_id: number;
  game_date: string;
  teams: TeamInfo[];
  players: PlayerSummary[];
};

