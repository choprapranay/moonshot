export type HeatmapPoint = {
  plate_x: number;
  plate_z: number;
  expected_value_diff: number;
};

export type SwingPoint = {
  plate_x: number;
  plate_z: number;
  pitch_type?: string | null;
  description?: string | null;
};

export type PlayerHeatmapResponse = {
  heatmap: HeatmapPoint[];
  swings: SwingPoint[];
};

