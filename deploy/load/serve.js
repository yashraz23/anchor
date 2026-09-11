// Load test for the vLLM generator.
//
// Constant arrival rate rather than a fixed number of virtual users: a VU loop
// measures how fast the server drains a queue it sets itself, which flatters a
// slow server. Arrival rate holds the offered load steady and lets latency move,
// which is what P99 is supposed to show.
import http from "k6/http";
import { check } from "k6";
import { Trend, Rate } from "k6/metrics";

const ttft = new Trend("time_to_response", true);
const failures = new Rate("failed_requests");

export const options = {
  scenarios: {
    sustained: {
      executor: "constant-arrival-rate",
      rate: Number(__ENV.RATE || 6),
      timeUnit: "1s",
      duration: __ENV.DURATION || "60s",
      preAllocatedVUs: 40,
      maxVUs: 120,
    },
  },
  thresholds: {
    // Recorded, not enforced as a gate: the point is to measure the knee, and
    // a failing threshold would abort the run that produces the number.
    http_req_duration: ["p(99)<600000"],
  },
};

// Prompts drawn from the same question distribution the golden set uses, so the
// generation lengths are representative rather than a single cached prompt.
const PROMPTS = [
  "What does max_num_seqs control in vLLM?",
  "How do I reduce GPU memory usage when serving a model?",
  "What is enforce_eager and when should I use it?",
  "Explain what the KV cache is used for.",
  "How does continuous batching differ from static batching?",
  "What does gpu_memory_utilization do?",
];

export default function () {
  const body = JSON.stringify({
    model: "anchor-generator",
    messages: [
      { role: "user", content: PROMPTS[Math.floor(Math.random() * PROMPTS.length)] },
    ],
    max_tokens: Number(__ENV.MAX_TOKENS || 64),
    temperature: 0,
  });

  const res = http.post(`${__ENV.TARGET || "http://127.0.0.1:8000"}/v1/chat/completions`, body, {
    headers: { "Content-Type": "application/json" },
    timeout: "120s",
  });

  ttft.add(res.timings.duration);
  failures.add(res.status !== 200);
  check(res, { "status 200": (r) => r.status === 200 });
}
