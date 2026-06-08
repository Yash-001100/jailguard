export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface AnalyzeResultData {
  sessionId: string;
  riskScore: number;
  label: "safe" | "suspicious" | "jailbreak";
  attackType: string | null;
  attackConfidence: number | null;
  flagged: boolean;
  latencyMs: number;
}

export declare class AnalyzeResult implements AnalyzeResultData {
  sessionId: string;
  riskScore: number;
  label: "safe" | "suspicious" | "jailbreak";
  attackType: string | null;
  attackConfidence: number | null;
  flagged: boolean;
  latencyMs: number;
}

export declare class Session {
  sessionId: string | null;
  lastResult: AnalyzeResult | null;
  send(userMessage: string, assistantReply?: string | null): Promise<AnalyzeResult>;
  clear(): Promise<void>;
}

export declare class JailGuard {
  constructor(opts: { apiKey: string; baseUrl?: string; timeoutMs?: number });
  analyze(messages: Message[], sessionId?: string | null): Promise<AnalyzeResult>;
  deleteSession(sessionId: string): Promise<void>;
  session(): Session;
}
