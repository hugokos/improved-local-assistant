# Dynamic KG - Golden Conversation Fixtures

These fixtures are intentionally **longer** (8-10 turns each) so your extraction pipeline and hybrid retriever get a realistic workout.

Each block is pure **JSONL** (one message per line, keys `role` and `content`). Copy a block to its own file (`fixture_1.jsonl`, etc.) when running tests.

---

## Fixture 1 - Survival Basics (trekking)

```jsonl
{"role": "user", "content": "I'm packing for a three-day trek. What essential items do I need for first aid?"}
{"role": "assistant", "content": "Bring sterile gauze, antiseptic wipes, tweezers, ibuprofen, and a mylar emergency blanket."}
{"role": "user", "content": "Why the emergency blanket?"}
{"role": "assistant", "content": "It reflects body heat and can treat early hypothermia if temperatures drop."}
{"role": "user", "content": "Got it. Anything for water purification?"}
{"role": "assistant", "content": "Yes—pack iodine tablets or a Sawyer squeeze filter; dehydration is the #1 trek-killer."}
{"role": "user", "content": "What about signaling for rescue?"}
{"role": "assistant", "content": "Add a whistle and a signal mirror; both weigh < 30 g and can be spotted kilometers away."}
{"role": "user", "content": "Cool. How often should I restock meds like ibuprofen?"}
{"role": "assistant", "content": "Replace sealed blister packs every two years; moisture ruins potency in the field."}
```

**Probe questions**

* "Which item treats hypothermia?" → emergency blanket.
* "Name two ways to purify water." → iodine tablets, Sawyer filter.
* "How far can a signal mirror be seen?" → kilometers away.

---

## Fixture 2 - Local-AI Setup Troubleshooting (Raspberry Pi 5)

```jsonl
{"role": "user", "content": "TinyLlama keeps running out of memory on my Raspberry Pi 5."}
{"role": "assistant", "content": "Quantise to Q4_K_M or switch to Phi-3-mini; both fit inside 4 GB."}
{"role": "user", "content": "How do I quantise?"}
{"role": "assistant", "content": "Run 'ollama pull phi3:mini && llama-quantize –preset Q4_K_M'."}
{"role": "user", "content": "Do I need AVX-512 for that?"}
{"role": "assistant", "content": "No—Pi 5 is ARM; just compile llama.cpp with make LLAMA_NO_MULTITHREAD=0 for NEON optimizations."}
{"role": "user", "content": "What thread count is ideal?"}
{"role": "assistant", "content": "Set OLLAMA_NUM_PREDICT to 256 and use 8 threads; that saturates all A76 cores."}
{"role": "user", "content": "How can I monitor memory while it runs?"}
{"role": "assistant", "content": "Use psutil.Process().memory_percent(); alert if it exceeds 90 percent."}
{"role": "user", "content": "Great—how do I enable keep-alive caching?"}
{"role": "assistant", "content": "Export OLLAMA_KEEP_ALIVE=30 so the model stays warm for quick successive calls."}
```

**Probe questions**

* "Which preset is recommended for quantisation?" → Q4_K_M.
* "Ideal thread count on Pi 5?" → 8 threads.
* "How long does keep-alive cache the model?" → 30 seconds.

---

## Fixture 3 - Startup Fund-raising Chat (YC prep)

```jsonl
{"role": "user", "content": "We need a compelling metric for our YC pitch."}
{"role": "assistant", "content": "Show week-over-week retention hitting 75 percent – elite for consumer apps."}
{"role": "user", "content": "Retention climbed after we added personalised jokes."}
{"role": "assistant", "content": "Note that feature in the slide: 'Humor engine → +18 pp retention'."}
{"role": "user", "content": "What about monthly active users?"}
{"role": "assistant", "content": "Highlight 45 k MAU with 12 k daily creators; both doubled post-personalisation."}
{"role": "user", "content": "Investors will ask about CAC."}
{"role": "assistant", "content": "Show CAC dropped to $1.12 after our TikTok funnel; payback under two weeks."}
{"role": "user", "content": "Nice. Any viral coefficient numbers?"}
{"role": "assistant", "content": "We hit k = 1.4 in June; anything above 1.3 signals organic flywheel."}
{"role": "user", "content": "Should I mention the Series-A soft circle?"}
{"role": "assistant", "content": "Yes—$4 M soft-committed from XYZ Capital validates traction."}
```

**Probe questions**

* "What retention figure goes in the deck?" → 75 percent.
* "Which feature boosted retention?" → humor engine / personalised jokes.
* "Current CAC?" → $1.12.
* "Viral coefficient value?" → 1.4.

---

### How to use these fixtures

1. **Extraction tests** – loop over each message (except probe questions) and assert expected triples exist. Because conversations are longer, you'll validate sliding-window chunking across multiple turns.
2. **Dynamic-graph integration** – replay full fixture through `update_dynamic_graph()`, then query with each probe question. Assert the hybrid retriever surfaces a node containing the expected phrase within top-k.
3. **Latency/resource checks** – measure extraction latency per turn and peak RSS; ensure metrics stay below your CI thresholds even with 10-turn inputs.