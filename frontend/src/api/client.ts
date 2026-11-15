/**
 * src/api/client.ts
 *
 * Responsibility: Typed API client using generated types from schema.d.ts.
 * Never hand-write duplicates of backend Pydantic models.
 */

import type { components } from './schema';

export type HealthResponse = components['schemas']['HealthResponse'];

export class ApiError extends Error {
  public status: number;
  public detail?: unknown;

  constructor(message: string, status: number, detail?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
    ...init,
  });

  if (!response.ok) {
    let errorDetail: unknown;
    try {
      errorDetail = await response.json();
    } catch {
      // response wasn't JSON
    }
    throw new ApiError(
      `Request failed with status ${response.status}: ${response.statusText}`,
      response.status,
      errorDetail
    );
  }

  return response.json() as Promise<T>;
}

export const api = {
  getHealth: async (): Promise<HealthResponse> => {
    return fetchJson<HealthResponse>('/api/health');
  },
};
