"use client";

import React, { useState } from "react";

interface PitchData {
  inning: number;
  inning_topbot: string;
  pitcher: number;
  pitch_name: string;
  plate_x: number;
  plate_z: number;
  batter: number;
  description: string;
  pitch_type: string;
  events: string;
  stand: string;
  p_throws: string;
  home_team: string;
  away_team: string;
  balls: number;
  strikes: number;
  zone: number;
  at_bat_number: number;
  pitch_number: number;
}

interface ApiResponse {
  success: boolean;
  game_id: string;
  total_pitches: number;
  data: PitchData[];
  available_columns?: string[];
  error?: string;
}

export default function Home() {
  const [gameId, setGameId] = useState("");
  const [pitchData, setPitchData] = useState<PitchData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [gameInfo, setGameInfo] = useState<{ game_id: string; total_pitches: number } | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!gameId.trim()) return;

    setLoading(true);
    setError("");
    setPitchData([]);
    setGameInfo(null);

    try {
      const response = await fetch(`http://localhost:5001/api/pitch-data/${gameId.trim()}`);
      const result: ApiResponse = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'Failed to fetch data');
      }

      if (result.success) {
        setPitchData(result.data);
        setGameInfo({ game_id: result.game_id, total_pitches: result.total_pitches });
      } else {
        throw new Error(result.error || 'Unknown error occurred');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getTeamSide = (inning_topbot: string) => {
    return inning_topbot === 'Top' ? 'Away' : 'Home';
  };

  const isSwingPitch = (description: string) => {
    const desc = description.toLowerCase();
    return desc.includes('swinging') || desc.includes('foul') || desc.includes('hit') ||
           desc.includes('single') || desc.includes('double') || desc.includes('triple') ||
           desc.includes('home_run') || desc.includes('groundout') || desc.includes('popout') ||
           desc.includes('flyout') || desc.includes('lineout');
  };

  const getPitchOutcome = (description: string, events: string) => {
    const desc = description.toLowerCase();
    const event = events ? events.toLowerCase() : '';
    
    if (desc.includes('swinging') || desc.includes('foul') || desc.includes('hit')) {
      if (desc.includes('swinging_strike') || desc.includes('swinging_strike_blocked')) {
        return 'Whiff';
      } else if (desc.includes('foul')) {
        return 'Foul';
      } else if (desc.includes('hit') || desc.includes('single') || desc.includes('double') || 
                 desc.includes('triple') || desc.includes('home_run') || desc.includes('groundout') ||
                 desc.includes('popout') || desc.includes('flyout') || desc.includes('lineout')) {
        return 'Ball in Play';
      } else {
        return 'Ball in Play';
      }
    }
    
    if (desc.includes('ball')) {
      return 'Ball';
    } else if (desc.includes('called_strike')) {
      return 'Called Strike';
    }
    
    return description;
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Moonshot</h1>
        </div>

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="flex gap-4 items-center justify-center mb-8">
          <input
            type="text"
            value={gameId}
            onChange={(e) => setGameId(e.target.value)}
            placeholder="Enter ID"
            className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent min-w-[300px]"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Loading..." : "Submit"}
          </button>
        </form>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6 max-w-2xl mx-auto">
            <strong>Error:</strong> {error}
          </div>
        )}

        {gameInfo && (
          <div className="bg-blue-100 border border-blue-400 text-blue-700 px-4 py-3 rounded mb-6 max-w-2xl mx-auto">
            <strong>Game ID:</strong> {gameInfo.game_id} | <strong>Total Pitches:</strong> {gameInfo.total_pitches}
          </div>
        )}

        {pitchData.length > 0 && (
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Inning</th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Team</th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Count</th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Pitcher</th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Pitch Type</th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Location (X, Z)</th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Batter</th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Swing/Take</th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Outcome</th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Batter Hand</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {pitchData.map((pitch, index) => {
                    const isSwing = isSwingPitch(pitch.description);
                    return (
                      <tr key={index} className="hover:bg-gray-50 transition-colors">
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{pitch.inning}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{getTeamSide(pitch.inning_topbot)}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {pitch.balls}-{pitch.strikes}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{pitch.pitcher}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{pitch.pitch_name || pitch.pitch_type || 'N/A'}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {pitch.plate_x !== null && pitch.plate_z !== null 
                            ? `(${pitch.plate_x.toFixed(2)}, ${pitch.plate_z.toFixed(2)})`
                            : 'N/A'
                          }
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{pitch.batter}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {isSwing ? 'SWING' : 'TAKE'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{getPitchOutcome(pitch.description, pitch.events)}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{pitch.stand}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}