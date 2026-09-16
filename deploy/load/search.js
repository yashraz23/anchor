// Load test for the retrieval endpoint, used to drive the HPA in
// deploy/k3s/50-api-hpa.yaml.
//
// Constant arrival rate, not a pool of virtual users, for the same reason as
// serve.js: a VU loop measures how fast the server drains a queue it set
// itself, which flatters a slow server. Here the server is deliberately slow --
// a search spends ~20 s in the cross-encoder on two cores -- so a VU loop would
// quietly reduce the offered load to whatever the server could already take and
// the autoscaler would never see a reason to act.
//
// /search rather than /ask: /ask calls Claude at about three cents a request,
// so a load test against it would bill real money per iteration and measure the
// API provider rather than this service.
import http from "k6/http";
import { check } from "k6";

const BASE = __ENV.BASE_URL || "http://127.0.0.1:18080";
const RATE = Number(__ENV.RATE || 12); // requests per minute
const MINUTES = Number(__ENV.MINUTES || 5);

// Real questions from the golden set. A single repeated query would be answered
// from Postgres' cache and from the same reranker batch shape every time, which
// would understate the work a real mix causes.
const QUERIES = [
  "how do I limit concurrent sequences",
  "what is prefix caching and when does it help",
  "how to serve a quantized model with vllm",
  "what does tensor parallel size actually do",
  "why is my throughput lower than expected",
  "how do I set the maximum context length",
  "what is continuous batching",
  "how do I enable automatic tool choice",
];

export const options = {
  scenarios: {
    search: {
      executor: "constant-arrival-rate",
      rate: RATE,
      timeUnit: "1m",
      duration: `${MINUTES}m`,
      // Headroom: at ~20 s per request and 12/min offered, about 4 are in
      // flight at steady state. Capping VUs too low would silently throttle the
      // offered rate and turn this back into a VU loop.
      preAllocatedVUs: 20,
      maxVUs: 60,
    },
  },
  thresholds: {
    // Deliberately loose. This run exists to make the autoscaler act, not to
    // assert a latency SLO on a laptop.
    http_req_failed: ["rate<0.10"],
  },
};

export default function () {
  const q = QUERIES[Math.floor(Math.random() * QUERIES.length)];
  const res = http.post(
    `${BASE}/search`,
    JSON.stringify({ query: q }),
    { headers: { "Content-Type": "application/json" }, timeout: "180s" },
  );
  check(res, {
    "status is 200": (r) => r.status === 200,
    "returned spans": (r) => {
      try {
        return JSON.parse(r.body).spans.length > 0;
      } catch (e) {
        return false;
      }
    },
  });
}
