import React from "react";
import { Zap, Search } from "lucide-react";
import type { OperationMode } from "../types";

interface Props {
  mode: OperationMode;
  onChange: (m: OperationMode) => void;
  disabled?: boolean;
}

export default function ModeSwitch({ mode, onChange, disabled }: Props) {
  return (
    <div className="flex rounded-xl border border-surface-border bg-surface-raised p-1 gap-1">
      <button
        onClick={() => !disabled && onChange("normal")}
        disabled={disabled}
        className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
          mode === "normal"
            ? "bg-accent-blue text-white shadow-lg"
            : "text-gray-400 hover:text-gray-200 hover:bg-surface"
        } ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
      >
        <Zap size={15} />
        Normal
      </button>
      <button
        onClick={() => !disabled && onChange("research")}
        disabled={disabled}
        className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
          mode === "research"
            ? "bg-accent-purple text-white shadow-lg"
            : "text-gray-400 hover:text-gray-200 hover:bg-surface"
        } ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
      >
        <Search size={15} />
        Research
      </button>
    </div>
  );
}
