import React, { useCallback, useRef, useState } from "react";
import { Send, Microscope, RotateCcw, ChevronRight } from "lucide-react";
import type {
  AnySSEEvent,
  JudgeResultEvent,
  IterationCompleteEvent,
  OperationMode,
  QueryState,
  ResultEvent,
  RouteEvent,
  SourceFoundEvent,
} from "./types";
import { startQuery, streamQuery } from "./api";
import ModeSwitch from "./components/ModeSwitch";
import DepthAccuracySliders from "./components/DepthAccuracySliders";
import ProgressFeed from "./components/ProgressFeed";
import ResultView from "./components/ResultView";

const INITIAL_STATE: QueryState = {
  status: "idle",
  progress: [],
  route: null,
  sources: [],
  streamingAnswer: "",
  finalResult: null,
  judgeEvents: [],
  iterationEvents: [],
  error: null,
  warning: null,
};

const SUGGESTIONS = [
  "What are the latest findings on Alzheimer's prevention?",
  "Explain how mRNA vaccines work",
  "What does the research say about remote work productivity?",
  "How does CRISPR gene editing work in humans?",
  "What are evidence-based treatments for anxiety?",
];

export default function App() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<OperationMode>("normal");
  const [depth, setDepth] = useState(3);
  const [accuracy, setAccuracy] = useState(3);
  const [queryState, setQueryState] = useState<QueryState>(INITIAL_STATE);
  const stopRef = useRef<(() => void) | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const isRunning = queryState.status !== "idle" && queryState.status !== "done" && queryState.status !== "error";

  const handleEvent = useCallback((event: AnySSEEvent) => {
    setQueryState((prev) => {
      switch (event.type) {
        case "progress":
          return {
            ...prev,
            status: event.step === "synthesize" ? "synthesizing" : event.step === "quorum" ? "judging" : "searching",
            progress: [...prev.progress, event as never],
          };
        case "route_complete": {
          // Update the last "route" progress entry in-place so RouteTag renders under it
          const re = event as RouteEvent;
          const updatedProgress = [...prev.progress];
          const lastRouteIdx = updatedProgress.map(p => p.step).lastIndexOf("route");
          if (lastRouteIdx >= 0) {
            updatedProgress[lastRouteIdx] = { ...updatedProgress[lastRouteIdx] };
          }
          return { ...prev, status: "routing", route: re, progress: updatedProgress };
        }
        case "source_found":
          return {
            ...prev,
            sources: [...prev.sources, (event as SourceFoundEvent).source],
          };
        case "synthesis_start":
          return { ...prev, status: "synthesizing" };
        case "synthesis_chunk":
          return { ...prev, streamingAnswer: prev.streamingAnswer + (event as { chunk: string }).chunk };
        case "synthesis_complete": {
          const sc = event as { answer: string };
          return { ...prev, streamingAnswer: sc.answer };
        }
        case "judge_result":
          return { ...prev, judgeEvents: [...prev.judgeEvents, event as JudgeResultEvent] };
        case "iteration_complete":
          return { ...prev, iterationEvents: [...prev.iterationEvents, event as IterationCompleteEvent] };
        case "result": {
          const r = event as ResultEvent;
          return {
            ...prev,
            status: "done",
            finalResult: r,
            sources: r.sources,
            streamingAnswer: r.answer,
          };
        }
        case "warning":
          return { ...prev, warning: (event as { message: string }).message };
        case "error":
          return { ...prev, status: "error", error: (event as { message: string }).message };
        default:
          return prev;
      }
    });
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!query.trim() || isRunning) return;

    if (stopRef.current) stopRef.current();
    setQueryState({ ...INITIAL_STATE, status: "routing" });

    try {
      const queryId = await startQuery(query.trim(), mode, depth, accuracy);
      const stop = streamQuery(queryId, handleEvent, () => {}, (err) => {
        setQueryState((prev) => ({ ...prev, status: "error", error: err }));
      });
      stopRef.current = stop;
    } catch (e) {
      setQueryState((prev) => ({ ...prev, status: "error", error: String(e) }));
    }
  }, [query, mode, depth, accuracy, isRunning, handleEvent]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleReset = () => {
    if (stopRef.current) stopRef.current();
    setQueryState(INITIAL_STATE);
    setQuery("");
    textareaRef.current?.focus();
  };

  const hasResult = queryState.status === "done" || queryState.streamingAnswer;

  return (
    <div className="min-h-screen bg-surface flex flex-col">
      {/* Header */}
      <header className="border-b border-surface-border bg-surface-raised/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <Microscope size={20} className="text-accent-blue" />
            <span className="font-semibold text-white tracking-tight">Research Assistant</span>
          </div>
          {(hasResult || isRunning) && (
            <button
              onClick={handleReset}
              className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors"
            >
              <RotateCcw size={13} />
              New query
            </button>
          )}
        </div>
      </header>

      <main className="flex-1 max-w-4xl mx-auto w-full px-4 py-8 flex flex-col gap-6">
        {/* Input section */}
        <div className="space-y-4">
          {/* Mode + sliders row */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 justify-between">
            <ModeSwitch mode={mode} onChange={setMode} disabled={isRunning} />
            {mode === "research" && (
              <div className="flex-1 max-w-sm w-full">
                <DepthAccuracySliders
                  depth={depth}
                  accuracy={accuracy}
                  onDepthChange={setDepth}
                  onAccuracyChange={setAccuracy}
                  disabled={isRunning}
                />
              </div>
            )}
          </div>

          {/* Query box */}
          <div className="relative">
            <textarea
              ref={textareaRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                mode === "normal"
                  ? "Ask anything — fast web search + synthesis..."
                  : "Ask a research question — multi-source retrieval + validation..."
              }
              disabled={isRunning}
              rows={3}
              className="w-full bg-surface-raised border border-surface-border rounded-xl px-4 py-3 pr-14 text-gray-100 placeholder-gray-600 focus:outline-none focus:border-accent-blue/60 focus:ring-1 focus:ring-accent-blue/30 resize-none text-sm leading-relaxed transition-colors scrollbar-thin disabled:opacity-60"
            />
            <button
              onClick={handleSubmit}
              disabled={!query.trim() || isRunning}
              className={`absolute bottom-3 right-3 p-2 rounded-lg transition-all ${
                query.trim() && !isRunning
                  ? mode === "research"
                    ? "bg-accent-purple text-white hover:bg-purple-500 shadow-lg"
                    : "bg-accent-blue text-white hover:bg-blue-500 shadow-lg"
                  : "bg-surface-border text-gray-600 cursor-not-allowed"
              }`}
            >
              <Send size={15} />
            </button>
          </div>

          <p className="text-xs text-gray-600 text-right">
            {isRunning ? "Running... " : ""}
            <kbd className="text-gray-700">⌘</kbd>+<kbd className="text-gray-700">Enter</kbd> to send
          </p>
        </div>

        {/* Suggestions (idle only) */}
        {queryState.status === "idle" && !query && (
          <div className="space-y-2 fade-in">
            <p className="text-xs text-gray-600 font-medium uppercase tracking-wider">Try asking</p>
            <div className="flex flex-wrap gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => setQuery(s)}
                  className="flex items-center gap-1.5 text-xs bg-surface-raised border border-surface-border text-gray-400 hover:text-gray-200 hover:border-gray-600 rounded-lg px-3 py-1.5 transition-colors"
                >
                  <ChevronRight size={11} />
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Progress feed */}
        {(isRunning || queryState.progress.length > 0) && queryState.status !== "done" && (
          <ProgressFeed state={queryState} />
        )}

        {/* Setup warning (Ollama not running) */}
        {queryState.warning && (
          <div className="bg-amber-900/20 border border-amber-700/40 rounded-xl px-4 py-3 text-sm text-amber-200 fade-in">
            <strong className="text-amber-300">Setup required:</strong>{" "}
            {queryState.warning}
          </div>
        )}

        {/* Error */}
        {queryState.status === "error" && queryState.error && (
          <div className="bg-red-900/20 border border-red-700/40 rounded-xl px-4 py-3 text-sm text-red-300 fade-in">
            <strong>Error:</strong> {queryState.error}
          </div>
        )}

        {/* Result */}
        <ResultView state={queryState} />
      </main>

      <footer className="border-t border-surface-border py-3 text-center text-xs text-gray-700">
        Local-first · Ollama default · Powered by open-access sources
      </footer>
    </div>
  );
}
