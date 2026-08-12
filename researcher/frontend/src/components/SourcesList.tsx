import React, { useState } from "react";
import { ExternalLink, BookOpen, Globe, FileText, ChevronDown, ChevronUp } from "lucide-react";
import type { Source } from "../types";

const TYPE_CONFIG = {
  "peer-reviewed": { label: "Peer reviewed", color: "text-accent-green", bg: "bg-green-900/30 border-green-700", icon: <BookOpen size={12} /> },
  preprint: { label: "Preprint", color: "text-amber-400", bg: "bg-amber-900/30 border-amber-700", icon: <FileText size={12} /> },
  article: { label: "Article", color: "text-accent-blue", bg: "bg-blue-900/30 border-blue-700", icon: <FileText size={12} /> },
  web: { label: "Web", color: "text-gray-400", bg: "bg-surface border-surface-border", icon: <Globe size={12} /> },
};

function SourceCard({ src }: { src: Source }) {
  const [expanded, setExpanded] = useState(false);
  const cfg = TYPE_CONFIG[src.type] || TYPE_CONFIG.web;

  return (
    <div className="bg-surface-raised border border-surface-border rounded-lg p-3 fade-in">
      <div className="flex items-start gap-2">
        <span className="mt-0.5 text-gray-500 flex-shrink-0">{cfg.icon}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <span className="text-sm text-gray-200 font-medium leading-snug line-clamp-2">
              {src.title || "Untitled"}
            </span>
            <span className={`text-xs px-1.5 py-0.5 rounded border flex-shrink-0 ${cfg.bg} ${cfg.color}`}>
              {cfg.label}
            </span>
          </div>

          <div className="flex items-center gap-2 mt-1">
            <span className="text-xs text-gray-600 font-mono truncate">{src.id}</span>
            {src.origin.length > 0 && (
              <div className="flex gap-1">
                {src.origin.map((o) => (
                  <span key={o} className="text-xs text-gray-500 bg-surface px-1 rounded">{o}</span>
                ))}
              </div>
            )}
          </div>

          {src.text_preview && (
            <div className="mt-2">
              <p className={`text-xs text-gray-400 leading-relaxed ${expanded ? "" : "line-clamp-2"}`}>
                {src.text_preview}
              </p>
              {src.text_preview.length > 200 && (
                <button
                  onClick={() => setExpanded(!expanded)}
                  className="flex items-center gap-0.5 text-xs text-gray-600 hover:text-gray-400 mt-0.5 transition-colors"
                >
                  {expanded ? <><ChevronUp size={10} /> Less</> : <><ChevronDown size={10} /> More</>}
                </button>
              )}
            </div>
          )}

          {src.url && (
            <a
              href={src.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-xs text-accent-blue hover:underline mt-1.5 truncate"
            >
              <ExternalLink size={10} />
              <span className="truncate">{src.url}</span>
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

export default function SourcesList({ sources }: { sources: Source[] }) {
  const [showAll, setShowAll] = useState(false);
  const display = showAll ? sources : sources.slice(0, 6);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
          Sources · {sources.length}
        </span>
        <div className="flex gap-2 text-xs text-gray-600">
          {(["peer-reviewed", "preprint", "article", "web"] as const).map((t) => {
            const n = sources.filter((s) => s.type === t).length;
            return n > 0 ? (
              <span key={t} className={TYPE_CONFIG[t].color}>
                {n} {t === "peer-reviewed" ? "peer-rev" : t}
              </span>
            ) : null;
          })}
        </div>
      </div>

      {display.map((src) => (
        <SourceCard key={src.id} src={src} />
      ))}

      {sources.length > 6 && (
        <button
          onClick={() => setShowAll(!showAll)}
          className="w-full text-xs text-gray-500 hover:text-gray-300 py-2 border border-surface-border rounded-lg transition-colors"
        >
          {showAll ? "Show less" : `Show ${sources.length - 6} more sources`}
        </button>
      )}
    </div>
  );
}
