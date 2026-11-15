import React from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  Cpu,
  CheckCircle2,
  Database,
  Layers,
  RefreshCw,
  ServerCrash,
  Sparkles,
} from 'lucide-react';
import { api, type HealthResponse } from '@/api/client';

export const HealthPage: React.FC = () => {
  const {
    data: health,
    isLoading,
    isError,
    error,
    refetch,
    isFetching,
    dataUpdatedAt,
  } = useQuery<HealthResponse, Error>({
    queryKey: ['health'],
    queryFn: api.getHealth,
    refetchInterval: 10_000, // Poll every 10 seconds per spec
    retry: 1,
  });

  const lastUpdated = dataUpdatedAt ? new Date(dataUpdatedAt).toLocaleTimeString() : null;

  return (
    <div className="max-w-4xl mx-auto py-8 px-4 sm:px-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">System Status & Provider</h1>
          <p className="text-sm text-slate-500 mt-1">
            Real-time status of backend agent engine, active LLM provider, and embeddings.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {lastUpdated && (
            <span className="text-xs text-slate-400">
              Updated {lastUpdated}
            </span>
          )}
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded-md hover:bg-slate-50 transition shadow-sm disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isFetching ? 'animate-spin text-emerald-600' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="bg-white border border-slate-200 rounded-xl p-8 shadow-sm text-center">
          <div className="inline-flex p-3 bg-slate-100 rounded-full text-slate-400 animate-pulse mb-3">
            <Activity className="h-6 w-6" />
          </div>
          <h3 className="text-base font-semibold text-slate-800">Connecting to Backend...</h3>
          <p className="text-sm text-slate-500 mt-1">Checking FastAPI server on /api/health</p>
        </div>
      )}

      {/* Graceful Error State — Backend Down */}
      {isError && (
        <div className="bg-white border border-red-200 rounded-xl p-8 shadow-sm">
          <div className="flex items-start gap-4">
            <div className="p-3 bg-red-50 text-red-600 rounded-lg shrink-0">
              <ServerCrash className="h-6 w-6" />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-red-100 text-red-800">
                  <span className="h-2 w-2 rounded-full bg-red-500"></span>
                  Backend Unreachable
                </span>
                <span className="text-xs text-slate-400">FastAPI Port 8000</span>
              </div>
              <h2 className="text-lg font-bold text-slate-900 mt-2">Cannot connect to Support Triage API</h2>
              <p className="text-sm text-slate-600 mt-1">
                The frontend could not reach <code className="px-1.5 py-0.5 bg-slate-100 text-slate-800 rounded text-xs font-mono">/api/health</code>.
                Make sure the backend server is running.
              </p>

              <div className="mt-4 p-3 bg-slate-900 rounded-lg text-xs font-mono text-slate-200 flex items-center justify-between">
                <span>uvicorn app.main:app --reload</span>
                <span className="text-slate-400 text-[10px]">backend terminal</span>
              </div>

              {error && (
                <div className="mt-3 text-xs text-red-600 font-mono bg-red-50 p-2 rounded border border-red-100">
                  Error details: {error.message}
                </div>
              )}

              <div className="mt-5">
                <button
                  onClick={() => refetch()}
                  className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-red-600 rounded-lg hover:bg-red-700 transition shadow-sm"
                >
                  <RefreshCw className="h-4 w-4" />
                  Try Again
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Healthy Live State */}
      {!isLoading && !isError && health && (
        <div className="space-y-6">
          {/* Main Status Banner */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-emerald-50 text-emerald-600 rounded-xl">
                <CheckCircle2 className="h-7 w-7" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800">
                    <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
                    Operational
                  </span>
                  <span className="text-xs text-slate-400 font-mono">v{health.version}</span>
                </div>
                <h2 className="text-xl font-bold text-slate-900 mt-1">API Backend is Active</h2>
                <p className="text-xs text-slate-500">FastAPI backend answering on /api/health with active provider configs.</p>
              </div>
            </div>

            <div className="text-xs text-slate-500 bg-slate-50 px-3 py-2 rounded-lg border border-slate-200 sm:text-right">
              <div>Auto-polling: <span className="font-semibold text-emerald-700">Active</span></div>
              <div className="text-slate-400 text-[11px] mt-0.5">Every 10 seconds via TanStack Query</div>
            </div>
          </div>

          {/* Provider Details Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Active Provider Card */}
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Active Provider</span>
                <Cpu className="h-4 w-4 text-slate-400" />
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold text-slate-900 capitalize">{health.provider}</span>
                <span className="text-xs font-medium px-2 py-0.5 bg-blue-50 text-blue-700 rounded-full border border-blue-200">
                  {health.provider === 'ollama' ? 'Local $0' : 'Cloud'}
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-2">
                Configured via <code className="text-slate-700 font-mono">configs/app.yaml</code>. Zero code changes to switch.
              </p>
            </div>

            {/* Chat Model Card */}
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Chat / Reasoning Model</span>
                <Sparkles className="h-4 w-4 text-slate-400" />
              </div>
              <div className="text-lg font-bold text-slate-900 font-mono truncate" title={health.chat_model}>
                {health.chat_model}
              </div>
              <p className="text-xs text-slate-500 mt-2">
                Powers Triage, Resolver, and Critic agents in the LangGraph state machine.
              </p>
            </div>

            {/* Embed Model Card */}
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Embedding Model</span>
                <Database className="h-4 w-4 text-slate-400" />
              </div>
              <div className="text-lg font-bold text-slate-900 font-mono truncate" title={health.embed_model}>
                {health.embed_model}
              </div>
              <p className="text-xs text-slate-500 mt-2">
                Powers KB article vector indexing and hybrid semantic retrieval.
              </p>
            </div>
          </div>

          {/* Architecture Highlight Card */}
          <div className="bg-gradient-to-r from-slate-900 to-slate-800 text-white rounded-xl p-6 shadow-sm">
            <div className="flex items-start gap-3">
              <Layers className="h-5 w-5 text-emerald-400 shrink-0 mt-0.5" />
              <div>
                <h3 className="text-sm font-semibold text-white">Multi-Provider Strategy Pattern</h3>
                <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                  LangGraph agent nodes call our custom <code className="text-emerald-300 font-mono">BaseLLM</code> interface, not LangChain vendor packages.
                  Switching between local Ollama and Google Gemini is a single line change in <code className="text-emerald-300 font-mono">configs/app.yaml</code>.
                  Every draft passes through the Critic compliance review before human approval or auto-send.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
