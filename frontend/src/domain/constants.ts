export const PITCH_TYPE_NAMES: Record<string, string> = {
  FF: "Four-seam Fastball",
  FT: "Two-seam Fastball",
  FC: "Cutter",
  SI: "Sinker",
  SL: "Slider",
  CH: "Changeup",
  CU: "Curveball",
  KC: "Knuckle Curve",
  KN: "Knuckleball",
  SC: "Screwball",
  FO: "Forkball",
  PO: "Pitch Out",
  FS: "Split-finger Fastball",
  ST: "Sweeper",
  SV: "Slurve",
  FA: "Fastball",
  EP: "Eephus",
  IN: "Intentional Ball",
  UN: "Unknown",
};

export function getPitchTypeName(abbreviation: string | null | undefined): string {
  if (!abbreviation) return "—";
  return PITCH_TYPE_NAMES[abbreviation.toUpperCase()] || abbreviation;
}

