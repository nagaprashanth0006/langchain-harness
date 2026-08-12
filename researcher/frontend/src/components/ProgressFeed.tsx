import React from "react";
import {
  Loader2, CheckCircle2, AlertCircle, Globe, BookOpen,
  Brain, Scale, RefreshCw
} from "lucide-react";
import type {
  ProgressEvent, RouteEvent, JudgeResultEvent, IterationCompleteEvent, QueryState
} from "../types";

const STEP_ICONS: Record<string, React.ReactNode> = {
  init: <Loader2 size={14} className="animate-spin text-gray-400" />,
  route: <Brain size={14} className="text-accent-blue" />,
  search: <Globe size={14} className="text-accent-blue" />,
  dedup: <RefreshCw size={14} className="text-gray-400" />,
  enrich: <Globe size={14} className="text-accent-blue" />,
  retrieve: <BookOpen size={14} className="text-accent-purple" />,
  synthesize: <Brain size={14} className="text-accent-green" />,
  quorum: <Scale size={14} className="text-amber-400" />,
  refine: <RefreshCw size={14} className="text-accent-blue" />,
};

function RouteTag({ route }: { route: RouteEvent }) {
  const categoryColors: Record<string, string> = {
    empirical_factual: "bg-blue-900/50 text-blue-300 border-blue-700",
    procedural_reference: "bg-purple-900/50 text-purple-300 border-purple-700",
    interpretive_subjective: "bg-amber-900/50 text-amber-300 border-amber-700",
    personal_sensitive: "bg-red-900/50 text-red-300 border-red-700",
  };
  const colorClass = categoryColors[route.category] || "bg-surface-raised text-gray-400 border-surface-border";
  const label = route.category.replace(/_/g, " ");
  return (
    <div className="flex flex-wrap items-center gap-2 mt-1">
      <span className={`text-xs px-2 py-0.5 rounded border ${colorClass}`}>{label}</span>
      {route.sensitivity && (
        <span className="text-xs px-2 py-0.5 rounded border bg-red-900/30 text-red-400 border-red-700">
          sensitive
        </span>
      )}
      {route.eligible_sources.map((s) => (
        <span key={s} className="text-xs px-2 py-0.5 rounded border bg-surface text-gray-500 border-surface-border">
          {s}
        </span>
      ))}
    </div>
  );
}

function JudgeRow({ ev }: { ev: JudgeResultEvent }) {
  return (
    <div className="flex items-center gap-2 text-xs fade-in">
      {ev.passed
        ? <CheckCircle2 size={12} className="text-accent-green flex-shrink-0" />
        : <AlertCircle size={12} className="text-amber-400 flex-shrink-0" />}
      <span className="text-gray-400">Judge</span>
      <span className={`font-mono ${ev.passed ? "text-accent-green" : "text-amber-400"}`}>
        {(ev.score * 100).toFixed(0)}%
      </span>
      <span className="text-gray-600">·</span>
      <span className="text-gray-500 truncate max-w-xs">{ev.gaps[0] || "ok"}</span>
    </div>
  );
}

function IterRow({ ev }: { ev: IterationCompleteEvent }) {
  const q = (ev.quality * 100).toFixed(0);
  const color = ev.quality >= 0.75 ? "text-accent-green" : ev.quality >= 0.5 ? "text-amber-400" : "text-red-400";
  return (
    <div className="flex items-center gap-2 text-xs fade-in border-t border-surface-border pt-2 mt-1">
      <span className="text-gray-500">Iteration {ev.iteration}</span>
      <span className={`font-semibold ${color}`}>quality {q}%</span>
      {ev.done && (
        <span className="text-gray-500 italic">→ {ev.reason.replace(/_/g, " ")}</span>
      )}
    </div>
  );
}

export default function ProgressFeed({ state }: { state: QueryState }) {
  return (
    <div className="bg-surface-raised border border-surface-border rounded-xl p-4 space-y-2 text-sm fade-in">
      {state.progress.map((ev, i) => (
        <div key={i} className="flex items-start gap-2 fade-in">
          <span className="mt-0.5 flex-shrink-0">
            {STEP_ICONS[ev.step] || <Loader2 size={14} className="animate-spin text-gray-500" />}
          </span>
          <div>
            <span className="text-gray-300">{ev.message}</span>
            {ev.step === "route" && state.route && <RouteTag route={state.route} />}
          </div>
        </div>
      ))}

      {state.judgeEvents.map((ev, i) => (
        <JudgeRow key={`j${i}`} ev={ev} />
      ))}

      {state.iterationEvents.map((ev, i) => (
        <IterRow key={`it${i}`} ev={ev} />
      ))}

      {state.status !== "done" && state.status !== "error" && state.status !== "idle" && (
        <div className="flex gap-1 pt-1">
          <span className="thinking-dot" />
          <span className="thinking-dot" />
          <span className="thinking-dot" />
        </div>
      )}
    </div>
  );
}
