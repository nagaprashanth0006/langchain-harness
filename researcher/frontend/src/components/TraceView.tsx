import React, { useState } from "react";
import { ChevronDown, ChevronRight, CheckCircle2, XCircle, AlertTriangle } from "lucide-react";
import type { ResultEvent } from "../types";

interface Props {
  result: ResultEvent;
}

function QualityBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = value >= 0.75 ? "bg-accent-green" : value >= 0.5 ? "bg-amber-400" : "bg-red-400";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-surface-border rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-gray-300 w-8 text-right">{pct}%</span>
    </div>
  );
}

function IterationRow({ iter }: { iter: NonNullable<ResultEvent["iterations"]>[number] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-surface-border rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3 py-2 bg-surface-raised hover:bg-surface transition-colors text-sm"
      >
        <div className="flex items-center gap-2">
          {iter.quorum_passed
            ? <CheckCircle2 size={14} className="text-accent-green" />
            : <XCircle size={14} className="text-amber-400" />}
          <span className="text-gray-300 font-medium">Iteration {iter.iteration}</span>
          <span className="text-gray-500 text-xs">{iter.sources_found} sources</span>
        </div>
        <div className="flex items-center gap-3">
          <QualityBar value={iter.quality} />
          {open ? <ChevronDown size={14} className="text-gray-500" /> : <ChevronRight size={14} className="text-gray-500" />}
        </div>
      </button>
      {open && iter.gap_list.length > 0 && (
        <div className="px-3 py-2 bg-surface border-t border-surface-border">
          <span className="text-xs text-gray-500 font-semibold uppercase tracking-wider">Gaps identified</span>
          <ul className="mt-1 space-y-0.5">
            {iter.gap_list.map((g, i) => (
              <li key={i} className="text-xs text-gray-400 flex gap-1.5">
                <span className="text-gray-600">·</span> {g}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function TraceView({ result }: Props) {
  const [open, setOpen] = useState(false);
  const hasIterations = result.iterations && result.iterations.length > 0;
  const hasContested = result.contested_positions && result.contested_positions.length > 0;

  if (!hasIterations && !hasContested) return null;

  return (
    <div className="border border-surface-border rounded-xl overflow-hidden fade-in">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 bg-surface-raised hover:bg-surface transition-colors"
      >
        <span className="text-sm font-semibold text-gray-300">Research trace</span>
        <div className="flex items-center gap-3">
          {result.final_confidence !== undefined && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">Confidence</span>
              <QualityBar value={result.final_confidence} />
            </div>
          )}
          {result.stop_reason && (
            <span className="text-xs text-gray-500 italic">{result.stop_reason.replace(/_/g, " ")}</span>
          )}
          {open ? <ChevronDown size={14} className="text-gray-500" /> : <ChevronRight size={14} className="text-gray-500" />}
        </div>
      </button>

      {open && (
        <div className="p-4 space-y-3 border-t border-surface-border bg-surface">
          {hasIterations && (
            <div className="space-y-2">
              <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Iterations</span>
              {result.iterations!.map((it) => (
                <IterationRow key={it.iteration} iter={it} />
              ))}
            </div>
          )}

          {hasContested && (
            <div className="space-y-2">
              <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1.5">
                <AlertTriangle size={12} className="text-amber-400" />
                Contested evidence
              </span>
              {result.contested_positions!.map((pos, i) => (
                <div key={i} className="bg-amber-900/20 border border-amber-700/40 rounded-lg px-3 py-2">
                  <span className="text-xs text-amber-300">{pos.aspect}</span>
                  <span className="text-xs text-gray-500 ml-2">· support: {pos.support}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
