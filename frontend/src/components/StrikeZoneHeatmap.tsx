import { forwardRef, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState } from "react";

type HeatmapPoint = {
  plate_x: number;
  plate_z: number;
  expected_value_diff: number;
};

type SwingPoint = {
  plate_x: number;
  plate_z: number;
  pitch_type?: string | null;
  description?: string | null;
};

type StrikeZoneHeatmapProps = {
  heatmap: HeatmapPoint[];
  swings: SwingPoint[];
  onSwingClick?: (swing: SwingPoint, index: number) => void;
  selectedSwingIndex?: number | null;
};

export type StrikeZoneHeatmapRef = {
  exportHeatmap: () => void;
};

const STRIKE_ZONE_HALF_WIDTH = 0.708; // 17 inches / 24
const STRIKE_ZONE_BOTTOM = 1.5;
const STRIKE_ZONE_TOP = 3.5;
const STRIKE_ZONE_MID = (STRIKE_ZONE_TOP + STRIKE_ZONE_BOTTOM) / 2;
const STRIKE_ZONE_WIDTH = STRIKE_ZONE_HALF_WIDTH * 2;
const STRIKE_ZONE_HEIGHT = STRIKE_ZONE_TOP - STRIKE_ZONE_BOTTOM;

const HORIZONTAL_EXTENT = STRIKE_ZONE_HALF_WIDTH + 1.0; // Natural extension outside strike zone
const VERTICAL_EXTENT_TOP = STRIKE_ZONE_TOP + 1.0;
const VERTICAL_EXTENT_BOTTOM = STRIKE_ZONE_BOTTOM - 0.8;

const HORIZONTAL_FALLOFF = STRIKE_ZONE_HALF_WIDTH + 0.3; // ~1 inch bleed
const VERTICAL_FALLOFF = STRIKE_ZONE_HEIGHT / 2 + 0.35;

export const StrikeZoneHeatmap = forwardRef<StrikeZoneHeatmapRef, StrikeZoneHeatmapProps>(({
  heatmap,
  swings,
  onSwingClick,
  selectedSwingIndex,
}, ref) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [dimensions, setDimensions] = useState<{ width: number; height: number }>({
    width: 0,
    height: 0,
  });
  const [hoveredSwingIndex, setHoveredSwingIndex] = useState<number | null>(null);

  useLayoutEffect(() => {
    const element = containerRef.current;
    if (!element) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setDimensions({ width, height });
        }
      }
    });

    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const { width, height } = dimensions;
    if (!canvas || width === 0 || height === 0) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const targetRatio = STRIKE_ZONE_WIDTH / STRIKE_ZONE_HEIGHT;
    let zoneHeight = height * 0.7;
    let zoneWidth = zoneHeight * targetRatio;
    if (zoneWidth > width * 0.75) {
      zoneWidth = width * 0.75;
      zoneHeight = zoneWidth / targetRatio;
    }

    const zoneCenterX = width / 2;
    const zoneCenterY = height / 2 - height * 0.05;
    const zoneX = zoneCenterX - zoneWidth / 2;
    const zoneY = zoneCenterY - zoneHeight / 2;
    const cornerRadius = 8;
    
    // Grid-based aggregation - coarser grid for bigger, flowing blobs
    const GRID_SIZE = 35; // Coarser grid = fewer, bigger blobs
    const grid: { sum: number; count: number }[][] = Array(GRID_SIZE)
      .fill(null)
      .map(() =>
        Array(GRID_SIZE)
          .fill(null)
          .map(() => ({ sum: 0, count: 0 }))
      );

    const drawRoundedRect = (
      context: CanvasRenderingContext2D,
      x: number,
      y: number,
      w: number,
      h: number,
      r: number
    ) => {
      const radius = Math.min(r, h / 2, w / 2);
      context.beginPath();
      context.moveTo(x + radius, y);
      context.lineTo(x + w - radius, y);
      context.quadraticCurveTo(x + w, y, x + w, y + radius);
      context.lineTo(x + w, y + h - radius);
      context.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
      context.lineTo(x + radius, y + h);
      context.quadraticCurveTo(x, y + h, x, y + h - radius);
      context.lineTo(x, y + radius);
      context.quadraticCurveTo(x, y, x + radius, y);
      context.closePath();
    };

    ctx.save();

    // Aggregate heatmap data into grid to reduce overlapping points
    heatmap.forEach((point) => {
      const adjustedPlateZ = point.plate_z + STRIKE_ZONE_MID;
      
      // Map to grid coordinates
      const normX = (point.plate_x + HORIZONTAL_EXTENT) / (2 * HORIZONTAL_EXTENT);
      const normZ = (adjustedPlateZ - VERTICAL_EXTENT_BOTTOM) / (VERTICAL_EXTENT_TOP - VERTICAL_EXTENT_BOTTOM);
      
      const gridX = Math.floor(normX * GRID_SIZE);
      const gridZ = Math.floor(normZ * GRID_SIZE);
      
      if (gridX >= 0 && gridX < GRID_SIZE && gridZ >= 0 && gridZ < GRID_SIZE) {
        grid[gridZ][gridX].sum += point.expected_value_diff;
        grid[gridZ][gridX].count += 1;
      }
    });
    
    // Helper functions
    const clampValue = (value: number, min: number, max: number) =>
      Math.max(min, Math.min(max, value));
    
    const plateXToPixel = (plateX: number) => {
      const clamped = clampValue(plateX, -HORIZONTAL_EXTENT, HORIZONTAL_EXTENT);
      return zoneCenterX + (clamped / STRIKE_ZONE_HALF_WIDTH) * (zoneWidth / 2);
    };
    
    const plateZToPixel = (plateZ: number) => {
      const clamped = clampValue(
        plateZ,
        VERTICAL_EXTENT_BOTTOM,
        VERTICAL_EXTENT_TOP
      );
      return (
        zoneCenterY -
        ((clamped - STRIKE_ZONE_MID) / (STRIKE_ZONE_HEIGHT / 2)) *
          (zoneHeight / 2)
      );
    };

    const baseRadius = Math.min(zoneWidth, zoneHeight) * 0.18;

    // Calculate max count for density adjustment
    let maxCount = 0;
    for (let gy = 0; gy < GRID_SIZE; gy++) {
      for (let gx = 0; gx < GRID_SIZE; gx++) {
        if (grid[gy][gx].count > maxCount) {
          maxCount = grid[gy][gx].count;
        }
      }
    }

    // Draw ALL cells (no aggressive filtering), with low alpha to handle massive data
    for (let gy = 0; gy < GRID_SIZE; gy++) {
      for (let gx = 0; gx < GRID_SIZE; gx++) {
        const cell = grid[gy][gx];
        if (cell.count === 0) continue;
        
        const avgEV = cell.sum / cell.count;
        // Only filter out extreme noise
        if (Math.abs(avgEV) < 0.008) continue;
        
        // Map grid back to plate coordinates (center of cell)
        const normX = (gx + 0.5) / GRID_SIZE;
        const normZ = (gy + 0.5) / GRID_SIZE;
        const plateX = normX * (2 * HORIZONTAL_EXTENT) - HORIZONTAL_EXTENT;
        const plateZ = normZ * (VERTICAL_EXTENT_TOP - VERTICAL_EXTENT_BOTTOM) + VERTICAL_EXTENT_BOTTOM;
        
        // Natural circular falloff - distance from strike zone center
        const distFromCenterX = Math.abs(plateX) / STRIKE_ZONE_HALF_WIDTH;
        const distFromCenterZ = Math.abs(plateZ - STRIKE_ZONE_MID) / (STRIKE_ZONE_HEIGHT / 2);
        const normalizedDist = Math.sqrt(distFromCenterX * distFromCenterX + distFromCenterZ * distFromCenterZ);
        
        // Very natural falloff - smooth probability-based filtering
        // Create organic, irregular edges like real data distribution
        const baseFalloff = 1.5; // Start falloff closer to strike zone
        if (normalizedDist > baseFalloff) {
          const excessDist = normalizedDist - baseFalloff;
          // Softer exponential falloff with more randomness
          // Use seeded random-like values based on coordinates to keep it deterministic
          const noise = Math.sin(gx * 2.3) * Math.cos(gy * 1.7) * 0.15; 
          const seed = Math.sin(gx * 12.9898 + gy * 78.233) * 43758.5453;
          const randomVal = seed - Math.floor(seed);
          
          const showProbability = Math.exp(-excessDist * 1.8) * (0.5 + randomVal * 0.5 + noise);
          if (randomVal > showProbability) continue;
        }
        
        // Additional scattered filtering for very natural look
        const seed2 = Math.cos(gx * 4.898 + gy * 32.23) * 23421.5453;
        const randomVal2 = seed2 - Math.floor(seed2);
        if (normalizedDist > 1.2 && randomVal2 > 0.85) continue;
        
        const px = plateXToPixel(plateX);
        const py = plateZToPixel(plateZ);
        
        // Scale intensity based on actual EV ranges (positive: 0-0.74, negative: 0 to -0.34)
        const isPositive = avgEV >= 0;
        
        // More sensitive normalization for better color variety
        let normalized;
        if (isPositive) {
          // For positive EV: 0.05 = weak, 0.2 = medium, 0.4+ = strong
          normalized = Math.min(avgEV / 0.5, 1);
        } else {
          // For negative EV: -0.05 = weak, -0.15 = medium, -0.3+ = strong
          normalized = Math.min(Math.abs(avgEV) / 0.35, 1);
        }
        
        // More responsive intensity curve - less smoothing for more variety
        const intensity = Math.pow(normalized, 0.4); // Lower exponent = more sensitivity
        
        // Radius scales with intensity - blue smaller, red slightly bigger
        const radiusMultiplier = isPositive ? 0.75 : 1.25;
        const radius = baseRadius * (0.8 + intensity * 0.5) * radiusMultiplier;
        
        // Adjust alpha based on data density
        const densityFactor = Math.sqrt(cell.count / maxCount);
        const densityPenalty = 1 - (densityFactor * 0.2);
        
        // Much wider alpha range for dramatic variety
        // Weak values: translucent, Medium values: semi-opaque, Strong values: very opaque
        const minAlpha = 0.03;  // Very translucent for weak signals
        const maxAlpha = 0.25;  // Much more opaque for strong signals
        const alpha = (minAlpha + intensity * maxAlpha) * densityPenalty; // 0.024 to 0.22
        
        // Create radial gradient
        const gradient = ctx.createRadialGradient(px, py, 0, px, py, radius);
        
        if (isPositive) {
          // Highly saturated deep blue
          gradient.addColorStop(0, `rgba(40, 110, 255, ${alpha})`);
          gradient.addColorStop(0.5, `rgba(40, 110, 255, ${alpha * 0.5})`);
          gradient.addColorStop(1, "rgba(40, 110, 255, 0)");
        } else {
          // Very vibrant red for negative EV
          gradient.addColorStop(0, `rgba(255, 50, 140, ${alpha})`);
          gradient.addColorStop(0.5, `rgba(255, 50, 140, ${alpha * 0.5})`);
          gradient.addColorStop(1, "rgba(255, 50, 140, 0)");
        }
        
        ctx.fillStyle = gradient;
        ctx.fillRect(px - radius, py - radius, radius * 2, radius * 2);
      }
    }

    ctx.restore();

    ctx.save();
    const haloRadius = Math.min(zoneWidth, zoneHeight) * 0.2;
    const haloGradient = ctx.createRadialGradient(
      zoneCenterX,
      zoneCenterY,
      haloRadius,
      zoneCenterX,
      zoneCenterY,
      haloRadius * 1.6
    );
    haloGradient.addColorStop(0, "rgba(90, 132, 255, 0.07)");
    haloGradient.addColorStop(1, "rgba(90, 132, 255, 0)");
    ctx.fillStyle = haloGradient;
    ctx.fillRect(
      zoneX - haloRadius * 0.8,
      zoneY - haloRadius * 0.7,
      zoneWidth + haloRadius * 1.6,
      zoneHeight + haloRadius * 1.4
    );
    ctx.restore();

    ctx.strokeStyle = "rgba(255, 255, 255, 0.78)";
    ctx.lineWidth = 2.75;
    drawRoundedRect(ctx, zoneX, zoneY, zoneWidth, zoneHeight, cornerRadius);
    ctx.stroke();
  }, [heatmap, dimensions]);
 
  const zoneLayout = useMemo(() => {
    const { width, height } = dimensions;
    if (width === 0 || height === 0) return null;

    const targetRatio = STRIKE_ZONE_WIDTH / STRIKE_ZONE_HEIGHT;
    let zoneHeight = height * 0.7;
    let zoneWidth = zoneHeight * targetRatio;
    if (zoneWidth > width * 0.75) {
      zoneWidth = width * 0.75;
      zoneHeight = zoneWidth / targetRatio;
    }

    const zoneCenterX = width / 2;
    const zoneCenterY = height / 2 - height * 0.05;

    const clampValue = (value: number, min: number, max: number) =>
      Math.max(min, Math.min(max, value));
    const plateXToPixel = (plateX: number) => {
      const clamped = clampValue(plateX, -HORIZONTAL_EXTENT, HORIZONTAL_EXTENT);
      return zoneCenterX + (clamped / STRIKE_ZONE_HALF_WIDTH) * (zoneWidth / 2);
    };
    const plateZToPixel = (plateZ: number) => {
      const clamped = clampValue(
        plateZ,
        VERTICAL_EXTENT_BOTTOM,
        VERTICAL_EXTENT_TOP
      );
      return (
        zoneCenterY -
        ((clamped - STRIKE_ZONE_MID) / (STRIKE_ZONE_HEIGHT / 2)) *
          (zoneHeight / 2)
      );
    };

    return { plateXToPixel, plateZToPixel };
  }, [dimensions]);

  // Expose export function to parent component
  useImperativeHandle(ref, () => ({
    exportHeatmap: () => {
      const canvas = canvasRef.current;
      if (!canvas || !zoneLayout) return;

      // Create a temporary canvas to include swing dots
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = canvas.width;
      tempCanvas.height = canvas.height;
      const tempCtx = tempCanvas.getContext('2d');
      if (!tempCtx) return;

      // Copy the original canvas (heatmap)
      tempCtx.drawImage(canvas, 0, 0);

      // Draw swing dots on top
      const dpr = window.devicePixelRatio || 1;
      swings.forEach((swing) => {
        const x = zoneLayout.plateXToPixel(swing.plate_x) * dpr;
        const y = zoneLayout.plateZToPixel(swing.plate_z) * dpr;

        // Draw outer glow circle
        tempCtx.beginPath();
        tempCtx.arc(x, y, 9 * dpr, 0, 2 * Math.PI);
        tempCtx.fillStyle = 'rgba(74, 108, 247, 0.25)';
        tempCtx.fill();

        // Draw inner white circle with blue border
        tempCtx.beginPath();
        tempCtx.arc(x, y, 6 * dpr, 0, 2 * Math.PI);
        tempCtx.fillStyle = '#f8f9ff';
        tempCtx.fill();
        tempCtx.strokeStyle = '#4a6cf7';
        tempCtx.lineWidth = 2 * dpr;
        tempCtx.stroke();
      });

      // Convert to blob and download
      tempCanvas.toBlob((blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        link.download = `heatmap-${timestamp}.png`;
        link.href = url;
        link.click();
        URL.revokeObjectURL(url);
      }, 'image/png');
    },
  }), [swings, zoneLayout]);

  return (
    <div ref={containerRef} className="relative flex h-full w-full flex-col overflow-hidden rounded-2xl">
      <canvas ref={canvasRef} className="absolute inset-0" />
      {zoneLayout && swings.map((swing, index) => {
        const x = zoneLayout.plateXToPixel(swing.plate_x);
        const y = zoneLayout.plateZToPixel(swing.plate_z);
        const isHovered = hoveredSwingIndex === index;
        const isSelected = selectedSwingIndex === index;
        const scale = isHovered ? 1.35 : isSelected ? 1.2 : 1;
        
        return (
          <button
            key={index}
            type="button"
            className="absolute cursor-pointer transition-transform duration-200 ease-out"
            style={{
              left: `${x}px`,
              top: `${y}px`,
              transform: `translate(-50%, -50%) scale(${scale})`,
            }}
            onMouseEnter={() => setHoveredSwingIndex(index)}
            onMouseLeave={() => setHoveredSwingIndex(null)}
            onClick={() => onSwingClick?.(swing, index)}
          >
            <div
              className="relative"
              style={{
                width: "18px",
                height: "18px",
              }}
            >
              <div
                className="absolute rounded-full"
                style={{
                  width: "18px",
                  height: "18px",
                  backgroundColor: "rgba(74, 108, 247, 0.25)",
                }}
              />
              <div
                className="absolute rounded-full"
                style={{
                  left: "50%",
                  top: "50%",
                  width: "12px",
                  height: "12px",
                  transform: "translate(-50%, -50%)",
                  backgroundColor: "#f8f9ff",
                  border: `${isSelected ? "2.5px" : "2px"} solid ${isSelected ? "#5b7fff" : "#4a6cf7"}`,
                  boxSizing: "border-box",
                }}
              />
            </div>
          </button>
        );
      })}
      <div className="pointer-events-none mt-auto flex justify-center gap-6 pb-4 text-[11px] tracking-wide text-[#9aa0d4]">
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-[#4a6cf7]" />
          <span>Positive EV (swing)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-[#ff627e]" />
          <span>Negative EV (take)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full border-2 border-[#4a6cf7] bg-white" />
          <span>Swings (selected game)</span>
        </div>
      </div>
    </div>
  );
});

StrikeZoneHeatmap.displayName = "StrikeZoneHeatmap";

