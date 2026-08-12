import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AlertTriangle, Clock } from "lucide-react";
import type { QueryState } from "../types";
import SourcesList from "./SourcesList";
import TraceView from "./TraceView";

interface Props {
  state: QueryState;
}

function SensitivityBanner() {
  return (
    <div className="flex items-start gap-2 bg-amber-900/20 border border-amber-700/40 rounded-lg px-4 py-3 text-sm fade-in">
      <AlertTriangle size={16} className="text-amber-400 flex-shrink-0 mt-0.5" />
      <span className="text-amber-200">
        This topic involves personal health, legal, financial, or mental-health considerations.
        The information provided is for general reference only — please consult qualified professionals
        for advice tailored to your situation.
      </span>
    </div>
  );
}

function StreamingCursor() {
  return (
    <span className="inline-block w-0.5 h-4 bg-accent-blue align-middle ml-0.5 animate-pulse" />
  );
}

export default function ResultView({ state }: Props) {
  const { streamingAnswer, finalResult, sources, status } = state;
  const showStreaming = streamingAnswer && status !== "done" && status !== "error";
  const answer = finalResult?.answer || streamingAnswer;

  if (!answer && !showStreaming && sources.length === 0) return null;

  return (
    <div className="space-y-4 fade-in">
      {finalResult?.sensitivity && <SensitivityBanner />}

      <div className="bg-surface-raised border border-surface-border rounded-xl p-6">
        <div className="prose-dark max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {answer}
          </ReactMarkdown>
          {showStreaming && <StreamingCursor />}
        </div>
      </div>

      {finalResult && (
        <div className="flex items-center gap-3 text-xs text-gray-600">
          <Clock size={12} />
          <span>{finalResult.elapsed.toFixed(1)}s</span>
          <span>·</span>
          <span>{sources.length} sources</span>
          {finalResult.final_confidence !== undefined && (
            <>
              <span>·</span>
              <span>confidence {(finalResult.final_confidence * 100).toFixed(0)}%</span>
            </>
          )}
        </div>
      )}

      {finalResult && <TraceView result={finalResult} />}

      {sources.length > 0 && <SourcesList sources={sources} />}
    </div>
  );
}
