import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

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

const STRIKE_ZONE_HALF_WIDTH = 0.708; // 17 inches / 24
const STRIKE_ZONE_BOTTOM = 1.5;
const STRIKE_ZONE_TOP = 3.5;
const STRIKE_ZONE_MID = (STRIKE_ZONE_TOP + STRIKE_ZONE_BOTTOM) / 2;
const STRIKE_ZONE_WIDTH = STRIKE_ZONE_HALF_WIDTH * 2;
const STRIKE_ZONE_HEIGHT = STRIKE_ZONE_TOP - STRIKE_ZONE_BOTTOM;

const HORIZONTAL_EXTENT = STRIKE_ZONE_HALF_WIDTH + 1.4; // allow balls way off the plate
const VERTICAL_EXTENT_TOP = STRIKE_ZONE_TOP + 1.4;
const VERTICAL_EXTENT_BOTTOM = STRIKE_ZONE_BOTTOM - 1.2;

const HORIZONTAL_FALLOFF = STRIKE_ZONE_HALF_WIDTH + 0.3; // ~1 inch bleed
const VERTICAL_FALLOFF = STRIKE_ZONE_HEIGHT / 2 + 0.35;

export function StrikeZoneHeatmap({
  heatmap,
  swings,
  onSwingClick,
  selectedSwingIndex,
}: StrikeZoneHeatmapProps) {
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

const baseRadius = Math.min(zoneWidth, zoneHeight) * 0.18;

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

    ctx.globalCompositeOperation = "lighter";

    const ellipticalDistance = (plateX: number, plateZ: number) => {
      const dx = Math.abs(plateX);
      const dz = Math.abs(plateZ - STRIKE_ZONE_MID);
      const nx = dx / HORIZONTAL_FALLOFF;
      const nz = dz / VERTICAL_FALLOFF;
      return Math.sqrt(nx * nx + nz * nz);
    };

    heatmap.forEach((point) => {
      const adjustedPlateZ = point.plate_z + STRIKE_ZONE_MID;
      const weighting = (() => {
        const dist = ellipticalDistance(point.plate_x, adjustedPlateZ);
        if (dist <= 1) return 1;
        return Math.exp(-(dist - 1) * 2.2);
      })();

      const x = plateXToPixel(point.plate_x);
      const y = plateZToPixel(adjustedPlateZ);
      const intensity = Math.min(
        Math.abs(point.expected_value_diff) / 0.16,
        1
      );
      const alpha = (0.12 + intensity * 0.28) * weighting;
      if (alpha <= 0.02) {
        return;
      }
      const isPositive = point.expected_value_diff >= 0;
      const radius =
        baseRadius *
        (0.9 + intensity * 0.4) *
        (0.85 + weighting * 0.3);
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);

      if (isPositive) {
        gradient.addColorStop(0, `rgba(96, 160, 255, ${alpha})`);
        gradient.addColorStop(1, "rgba(96, 160, 255, 0)");
      } else {
        gradient.addColorStop(0, `rgba(247, 102, 136, ${alpha})`);
        gradient.addColorStop(1, "rgba(247, 102, 136, 0)");
      }

      ctx.fillStyle = gradient;
      ctx.fillRect(x - radius, y - radius, radius * 2, radius * 2);
    });

    ctx.restore();

    ctx.save();
    const haloRadius = Math.min(zoneWidth, zoneHeight) * 0.2;
    const haloGradient = ctx.createRadialGradient(
      zoneCenterX,
      zoneCenterY,
      haloRadius,
      zoneCenterX,
      zoneCenterY,
      haloRadius * 3.4
    );
    haloGradient.addColorStop(0, "rgba(90, 132, 255, 0.07)");
    haloGradient.addColorStop(1, "rgba(90, 132, 255, 0)");
    ctx.fillStyle = haloGradient;
    ctx.fillRect(
      zoneX - haloRadius * 2.2,
      zoneY - haloRadius * 2,
      zoneWidth + haloRadius * 4.4,
      zoneHeight + haloRadius * 4
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
          <span className="h-3 w-3 rounded-full border-2 border-[#4a6cf7] bg-white" />
          <span>Swings (selected game)</span>
        </div>
      </div>
    </div>
  );
}

