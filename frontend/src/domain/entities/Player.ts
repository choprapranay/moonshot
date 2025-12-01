export type PlayerStats = {
  pitches_seen: number;
  swing_percentage: number;
  take_percentage: number;
  whiff_percentage: number;
  contact_percentage: number;
  average_velocity: number | null;
  batter_handedness?: string | null;
};

export type PlayerSummary = {
  player_id: number;
  player_name: string;
  team: string;
  headshot_url?: string | null;
  stats: PlayerStats;
  impact_zone_delta?: number | null;
};

