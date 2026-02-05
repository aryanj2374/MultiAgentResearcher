export type Role = "user" | "assistant";

export type MessageMeta = {
  response?: BackendResponse;
  error?: string;
  typing?: boolean;
  request?: { question: string };
};

export type Message = {
  id: string;
  role: Role;
  content: string;
  createdAt: string;
  meta?: MessageMeta;
};

export type Conversation = {
  id: string;
  title: string;
  createdAt: string;
  messages: Message[];
};

export type Paper = {
  paper_id: string;
  title: string;
  authors: string[];
  year?: number | null;
  venue?: string | null;
  url?: string | null;
  abstract?: string | null;
};

export type StudyExtraction = {
  paper_id: string;
  claim_summary: string;
  study_type: string;
  population?: string | null;
  sample_size?: number | null;
  intervention_exposure?: string | null;
  comparison?: string | null;
  outcomes?: string | null;
  effect_direction?: string | null;
  effect_size_text?: string | null;
  key_snippet: string;
  limitations: string[];
  apa_citation: string;
  url?: string | null;
};

export type Critique = {
  paper_id: string;
  risk_of_bias: string;
  rationale: string[];
  red_flags: string[];
};

export type Synthesis = {
  final_answer: string[];
  evidence_consensus: string;
  top_limitations_overall: string[];
  confidence_score: number;
  confidence_rationale: string[];
  citations_used: string[];
};

export type Verification = {
  passed: boolean;
  issues: string[];
  revised_synthesis?: Synthesis | null;
};

export type BackendResponse = {
  run_id: string;
  question: string;
  papers: Paper[];
  extractions: StudyExtraction[];
  critiques: Critique[];
  synthesis: Synthesis;
  verification: Verification;
  logs?: Record<string, unknown> | null;
};

export type Theme = "dark" | "light";

// Agent progress tracking types
export type AgentStatus = "pending" | "running" | "completed" | "failed";

export type AgentName = "retriever" | "extractor" | "critic" | "synthesizer" | "referee";

export type AgentProgress = Record<AgentName, AgentStatus>;

export type ProgressEvent = {
  type: "progress" | "result" | "error";
  agent?: AgentName;
  status?: AgentStatus;
  message?: string;
  data?: BackendResponse;
};

export const AGENT_NAMES: AgentName[] = ["retriever", "extractor", "critic", "synthesizer", "referee"];

export const AGENT_LABELS: Record<AgentName, string> = {
  retriever: "Retriever",
  extractor: "Extractor",
  critic: "Critic",
  synthesizer: "Synthesizer",
  referee: "Referee",
};

export function createInitialProgress(): AgentProgress {
  return {
    retriever: "pending",
    extractor: "pending",
    critic: "pending",
    synthesizer: "pending",
    referee: "pending",
  };
}
