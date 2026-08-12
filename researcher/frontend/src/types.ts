export type OperationMode = "normal" | "research";

export type QueryCategory =
  | "empirical_factual"
  | "procedural_reference"
  | "interpretive_subjective"
  | "personal_sensitive";

export interface Source {
  id: string;
  title: string;
  url: string;
  type: "peer-reviewed" | "preprint" | "article" | "web";
  origin: string[];
  text_preview: string;
  retrieved_at: number;
}

export interface IterationResult {
  iteration: number;
  sources_found: number;
  new_sources: number;
  quality: number;
  gap_list: string[];
  quorum_passed: boolean;
}

export interface SSEEvent {
  type: string;
  [key: string]: unknown;
}

export interface ProgressEvent extends SSEEvent {
  type: "progress";
  step: string;
  message: string;
}

export interface RouteEvent extends SSEEvent {
  type: "route_complete";
  category: QueryCategory;
  secondary: QueryCategory | null;
  sensitivity: boolean;
  eligible_sources: string[];
  rubric: string;
  reasoning: string;
}

export interface SourceFoundEvent extends SSEEvent {
  type: "source_found";
  source: Source;
}

export interface SynthesisStartEvent extends SSEEvent {
  type: "synthesis_start";
}

export interface SynthesisChunkEvent extends SSEEvent {
  type: "synthesis_chunk";
  chunk: string;
}

export interface JudgeResultEvent extends SSEEvent {
  type: "judge_result";
  target: string;
  passed: boolean;
  score: number;
  agreement: number;
  gaps: string[];
}

export interface IterationCompleteEvent extends SSEEvent {
  type: "iteration_complete";
  iteration: number;
  quality: number;
  done: boolean;
  reason: string;
}

export interface ResultEvent extends SSEEvent {
  type: "result";
  answer: string;
  sources: Source[];
  elapsed: number;
  mode: OperationMode;
  sensitivity: boolean;
  category: QueryCategory;
  // research-mode extras
  iterations?: IterationResult[];
  final_confidence?: number;
  stop_reason?: string;
  contested_positions?: Array<{ aspect: string; support: string; contested: boolean }>;
}

export interface ErrorEvent extends SSEEvent {
  type: "error";
  message: string;
}

export interface KeepaliveEvent extends SSEEvent {
  type: "keepalive";
}

export interface WarningEvent extends SSEEvent {
  type: "warning";
  message: string;
}

export interface SynthesisCompleteEvent extends SSEEvent {
  type: "synthesis_complete";
  answer: string;
}

export type AnySSEEvent =
  | ProgressEvent
  | RouteEvent
  | SourceFoundEvent
  | SynthesisStartEvent
  | SynthesisChunkEvent
  | SynthesisCompleteEvent
  | JudgeResultEvent
  | IterationCompleteEvent
  | ResultEvent
  | ErrorEvent
  | KeepaliveEvent
  | WarningEvent;

export interface QueryState {
  status: "idle" | "routing" | "searching" | "synthesizing" | "judging" | "done" | "error";
  progress: ProgressEvent[];
  route: RouteEvent | null;
  sources: Source[];
  streamingAnswer: string;
  finalResult: ResultEvent | null;
  judgeEvents: JudgeResultEvent[];
  iterationEvents: IterationCompleteEvent[];
  error: string | null;
  warning: string | null;
}
