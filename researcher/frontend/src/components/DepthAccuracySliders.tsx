import React from "react";
import { Layers, Target } from "lucide-react";

interface Props {
  depth: number;
  accuracy: number;
  onDepthChange: (v: number) => void;
  onAccuracyChange: (v: number) => void;
  disabled?: boolean;
}

const DEPTH_LABELS = ["", "Shallow", "Quick", "Balanced", "Deep", "Exhaustive"];
const ACCURACY_LABELS = ["", "Draft", "Standard", "Thorough", "Strict", "Max"];

function Slider({
  label,
  icon,
  value,
  onChange,
  disabled,
  labels,
  color,
}: {
  label: string;
  icon: React.ReactNode;
  value: number;
  onChange: (v: number) => void;
  disabled?: boolean;
  labels: string[];
  color: string;
}) {
  const pct = ((value - 1) / 4) * 100;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-xs font-medium text-gray-400 uppercase tracking-wider">
          {icon}
          {label}
        </span>
        <span className={`text-xs font-semibold ${color}`}>{labels[value]}</span>
      </div>
      <div className="relative">
        <div className="h-1.5 rounded-full bg-surface-border overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${color === "text-accent-blue" ? "bg-accent-blue" : "bg-accent-purple"}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <input
          type="range"
          min={1}
          max={5}
          step={1}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))}
          className={`absolute inset-0 w-full opacity-0 h-full cursor-pointer slider-thumb ${disabled ? "cursor-not-allowed" : ""}`}
        />
        <div className="flex justify-between mt-1">
          {[1, 2, 3, 4, 5].map((v) => (
            <div
              key={v}
              className={`w-1 h-1 rounded-full transition-all ${
                v <= value ? (color === "text-accent-blue" ? "bg-accent-blue" : "bg-accent-purple") : "bg-surface-border"
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

export default function DepthAccuracySliders({ depth, accuracy, onDepthChange, onAccuracyChange, disabled }: Props) {
  return (
    <div className="flex gap-6 flex-wrap">
      <div className="flex-1 min-w-40">
        <Slider
          label="Depth"
          icon={<Layers size={12} />}
          value={depth}
          onChange={onDepthChange}
          disabled={disabled}
          labels={DEPTH_LABELS}
          color="text-accent-blue"
        />
      </div>
      <div className="flex-1 min-w-40">
        <Slider
          label="Accuracy"
          icon={<Target size={12} />}
          value={accuracy}
          onChange={onAccuracyChange}
          disabled={disabled}
          labels={ACCURACY_LABELS}
          color="text-accent-purple"
        />
      </div>
    </div>
  );
}
