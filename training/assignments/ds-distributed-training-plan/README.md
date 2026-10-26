# Plan the training of a 7B decoder on 64 GPUs

You get 8 nodes of 8 H100 80 GB GPUs (NVLink inside each node, 400 Gb/s InfiniBand per GPU between nodes) for 8 weeks, and a 7B-parameter decoder-only transformer is due at the end: 32 layers, d_model 4096, 32 attention heads with 8 KV heads, a 128K vocabulary, 4,096-token context, trained with AdamW in bf16 mixed precision on 1 trillion pre-tokenized tokens. The cluster's history shows about one node or network incident every 3 days. Plan the run: memory per device, parallelism, throughput and schedule, checkpointing, and failure handling.

Spend 30 minutes. Write your answer as if you were talking through it with an interviewer, in this order:

1. **Memory per device.** Parameters, gradients, master weights and optimizer state in mixed precision; activation memory per token per layer and what sequence length and micro-batch that allows; whether any of it fits on one GPU unsharded. Ask what you'd want to clarify.
2. **Parallelism.** Data parallel with FSDP or ZeRO versus tensor parallel inside a node versus pipeline across nodes. Pick a layout and give the micro-batch, the global batch in tokens, gradient accumulation, and the bytes moved per step versus the compute per step.
3. **Throughput and schedule.** Total FLOPs from 6ND, the MFU you assume and why, tokens per second, days to finish, and the learning-rate schedule (warmup, peak, decay) with the margin left in the 8 weeks.
4. **Checkpointing and data.** Checkpoint frequency from the failure rate, checkpoint size and write time, asynchronous saves, resumable data-loader state, and deterministic sharding so every token is seen once.
5. **Failure handling and monitoring.** How a hung or dead rank is detected, elastic restart, the loss-spike playbook, what you watch on the dashboard, and what you do if you are 20% behind at the end of week 3.

Then score yourself with the rubric, one point per item, or paste your answer into a Claude session and ask for a rubric-based score with specific feedback.

## What the interviewer listens for

Whether you reason from bytes per parameter and 6ND FLOPs to a plan with numbers, rather than naming frameworks. Whether the plan fits in 8 weeks with the failure rate included, and what you would cut if it did not. Whether you can explain what happens to a restart that resumes the model from a checkpoint but not the data loader from its position.
