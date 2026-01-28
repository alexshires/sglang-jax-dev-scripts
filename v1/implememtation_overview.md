SGLang Scoring API in JAX - Gemini Notes
Overview
1. SGLang Library Overview
1.1 Core Architecture
1.2 The /v1/score Endpoint
2. SGLang Translation to JAX (sgl-jax)
2.1 Directory Structure
2.2 Key Architectural Changes in JAX
3. Fundamentals of SGLang JAX Implementation
3.1 The HTTP Server
3.2 The Scheduler
3.3 The Model Runner (ModelRunner)
4. Implementation Strategy for /v1/score
5. References
Scoring API
Technical Deep Dive: SGLang Scoring API on GPU with FlashInfer
1. The Core Concept: Scoring as Specialized Prefill
2. Data Flow Architecture
3. FlashInfer CUDA Integration
Summary for JAX Reimplementation
Implementation
Step 1: Update TokenizerManager to Handle Scoring
Step 2: Modify Scheduler to Dispatch Scoring Requests
Step 3: Implement Logprob Extraction in ModelRunner
Step 4: Expose via HTTP Server
Step 5: (Advanced) Block-Diagonal Masking (Optional Optimization)

Overview
Technical Report: SGLang Architecture and JAX Implementation Fundamentals
Abstract
This report provides a comprehensive technical overview of the SGLang library, focusing on its architecture and its port to the JAX ecosystem (sgl-jax). It analyzes the fundamental components of the SGLang runtime (SRT), the translation of PyTorch mechanisms to JAX/Flax primitives, and the specific execution flows relevant to implementing the /v1/score API (scoring/evaluation without generation).

1. SGLang Library Overview
SGLang (Structured Generation Language) is a high-performance framework designed for Large Language Model (LLM) serving and complex inference tasks. Its core value proposition lies in RadixAttention, a technique that enables automatic and efficient reuse of Key-Value (KV) caches across requests sharing common prefixes.
1.1 Core Architecture
The SGLang Runtime (SRT) follows a multi-process architecture to decouple request handling, scheduling, and model execution:
Tokenizer Manager: Handles HTTP requests, tokenization, and detokenization. It manages the RadixCache indices on the CPU to decide which blocks can be reused.
Scheduler: Manages the request queue and schedules batches. In the JAX implementation, this component orchestrates the Tensor Parallel (TP) workers via ZeroMQ (ZMQ).
Model Worker (Executor): Runs the actual neural network forward pass on the GPU/TPU.
1.2 The /v1/score Endpoint
The /v1/score API is distinct from generation. It calculates the log probabilities of a specific suffix given a prefix (e.g., P(answer | question)). Unlike generation, it does not sample new tokens; it evaluates existing token sequences.
In the PyTorch implementation (sglang/srt/entrypoints/openai/serving_score.py), the flow bypasses the standard generation request object:
Python
#
async def _handle_non_streaming_request(self, ...):
    # ...
    scores = await self.tokenizer_manager.score_request(
        query=request.query,
        items=request.items,
        # ...
    )

2. SGLang Translation to JAX (sgl-jax)
The sgl-jax project adapts the SGLang architecture to run on Google TPUs (and AMD/NVIDIA GPUs via JAX). While the high-level architecture (HTTP -> Tokenizer -> Scheduler -> Model) remains consistent, the backend implementation differs significantly due to JAX's functional paradigm and compilation requirements.
2.1 Directory Structure
The JAX implementation resides in python/sgl_jax and mirrors the PyTorch structure:
srt/entrypoints/: FastAPI server setup.
srt/managers/: Scheduler and Tokenizer logic.
srt/model_executor/: JAX/Flax model definitions and runner.
srt/layers/: Custom layers (Attention, MoE) implemented in JAX/Pallas/Triton.
2.2 Key Architectural Changes in JAX
State Management (Flax NNX): sgl-jax utilizes flax.nnx for managing model state, allowing for a more PyTorch-like object-oriented style within JAX's functional constraints.
Compilation (JIT): The model forward pass is JIT-compiled using jax.jit. Static argument mapping (like model_state_def) is critical for performance.
Sharding (Mesh): Instead of PyTorch's dist communication primitives, sgl-jax relies on jax.sharding.Mesh and NamedSharding to handle Tensor Parallelism automatically via the compiler.

3. Fundamentals of SGLang JAX Implementation
To reimplement /v1/score, one must understand how sgl-jax schedules batches and executes models.
3.1 The HTTP Server
The JAX HTTP server (sgl_jax/srt/entrypoints/http_server.py) registers the /v1/score endpoint. Currently, it instantiates OpenAIServingScore:
Python
#
fast_api_app.state.openai_serving_score = OpenAIServingScore(_global_state.tokenizer_manager)

@app.post("/v1/score", dependencies=[Depends(validate_json_request)])
async def v1_score_request(request: ScoringRequest, raw_request: Request):
    return await raw_request.app.state.openai_serving_score.handle_request(request, raw_request)
Note: The existing OpenAIServingScore in JAX likely relies on tokenizer_manager.score_request, which may utilize generation logic internally or need specialized handling for efficiency.
3.2 The Scheduler
The Scheduler (sgl_jax/srt/managers/scheduler.py) runs an event loop that manages ScheduleBatch objects. It differentiates between Prefill (extend) and Decode phases.
For scoring, the request is essentially a prefill-only operation. The model processes the prompt + output tokens in one go, extracting logits, but does not transition the request to the decoding queue.
Critical component for Scoring Reimplementation: The Scheduler class processes requests. You likely need to ensure handle_generate_request or a new handle_score_request creates a Req object that flags the scheduler to return logprobs and finish immediately after the prefill phase.
Python
#
def run_batch(self, batch: ScheduleBatch) -> GenerationBatchResult:
    # ...
    logits_output, next_token_ids_device, cache_miss_count = (
        self.tp_worker.forward_batch_generation(
            model_worker_batch, sampling_metadata=None
        )
    )
3.3 The Model Runner (ModelRunner)
This is the heart of the JAX execution. Located in sgl_jax/srt/model_executor/model_runner.py, it manages the JIT-compiled forward function.
JIT Compilation Setup:
Python
#
@partial(
    jax.jit,
    donate_argnames=["token_to_kv_pool"],  # just donate KV cache
    static_argnames=["model_state_def"],
)
def jitted_run_model(
    model_def,
    model_state_def,
    model_state_leaves,
    forward_batch,
    token_to_kv_pool,
    logits_metadata,
):
    # ...
    model = nnx.merge(model_def, model_state)
    return model(forward_batch, token_to_kv_pool, logits_metadata)
For scoring, the forward_batch data structure passed to this function contains the input IDs and positions. The logits_metadata instructs the model which logits to gather.
Forward Pass Logic: The ModelRunner.forward method invokes the JIT function. For scoring:
Input: A batch of sequences (prompt + target).
Execution: Run the model forward pass (same as prefill).
Output: Instead of passing logits to a Sampler (which selects 1 token), the runner must return the log probabilities for the specific target IDs provided in the request.
The compute_logprobs helper (JIT-compiled) is available in ModelRunner:
Python
#
@partial(jax.jit, static_argnames=["mesh"])
def jitted_compute_logprobs(mesh, logits, next_tokens):
    return compute_logprobs(mesh, logits, next_tokens)
Note: While named next_tokens, in a scoring context, this argument would represent the "target tokens" for which we want to compute the score.
4. Implementation Strategy for /v1/score
To reimplement or optimize /v1/score in JAX, the following flow is recommended:
Request Handling: Ensure OpenAIServingScore constructs a Req object where return_logprob=True and logprob_start_len is set to the start of the "target" segment (or 0 for full sequence scoring).
Scheduling: Modify Scheduler to ensure that requests marked for scoring are treated as "prefill-only". Once the prefill computation (and logprob extraction) is done, the request should be marked as finished and not added to the decoding queue.
Model Execution:
Ensure ModelRunner extracts logits for all requested positions, not just the last token.
Verify LogitsProcessor or compute_logprobs can handle seq_len > 1 vectors (parallel logprob computation for the whole sequence).
5. References
Serving Score Entrypoint (PyTorch Reference): sglang/srt/entrypoints/openai/serving_score.py
JAX HTTP Server: sgl_jax/srt/entrypoints/http_server.py
JAX Scheduler: sgl_jax/srt/managers/scheduler.py
JAX Model Runner: sgl_jax/srt/model_executor/model_runner.py

Scoring API

Based on the provided codebase, specifically the flashinfer_backend.py and serving_score.py files, here is a deep technical deep dive into how the SGLang scoring API utilizes the FlashInfer backend on GPU.
Technical Deep Dive: SGLang Scoring API on GPU with FlashInfer
The SGLang /v1/score API is optimized to evaluate the probability of multiple "items" (candidate completions) given a "query" (prompt) efficiently. Unlike generation, which is a serial decoding process, scoring is essentially a parallel "prefill" operation.
The FlashInferAttnBackend in SGLang implements a specialized mechanism called Multi-Item Scoring to handle this efficiently on CUDA GPUs.
1. The Core Concept: Scoring as Specialized Prefill
In a standard LLM generation, "prefill" processes the prompt in parallel. Scoring a sequence (Prompt + Completion) is mathematically identical to a prefill pass where you compute the logits for the "Completion" tokens but do not sample new ones.
SGLang optimizes this by packing multiple candidate completions for a single prompt into one large sequence (or batch) while using custom attention masking to ensuring candidates do not attend to each other.
2. Data Flow Architecture
A. Entry Point (OpenAIServingScore)
The request enters via serving_score.py.
Crucially, it calls tokenizer_manager.score_request.
It does not create a standard decoding loop. It creates a Req object marked for scoring (likely effectively treated as a prefill-only job).
B. The MultiItemScoringParams Structure
In python/sglang/srt/layers/attention/flashinfer_backend.py, SGLang defines a specific data class to manage this on the GPU:
Python
@dataclass
class MultiItemScoringParams:
    # A uint32 1D tensor: prefix length of each prompt (Query length)
    prefix_len_ptr: Optional[torch.Tensor] = None
    
    # A uint16 1D tensor: relative token positions reset for each item
    token_pos_in_items_ptr: Optional[torch.Tensor] = None
    
    # Zero padding length for batching
    token_pos_in_items_len: int = 0
    
    # Max length of any item in the batch (for kernel limits)
    max_item_len_ptr: Optional[torch.Tensor] = None
C. The _process_multi_item_scoring Method
The backend includes a specialized pre-processing step _process_multi_item_scoring that transforms a flat input batch into a structure FlashInfer can understand.
Delimiter Handling: It scans the input_ids for a special delimiter token.
Format: Query <delimiter> Item1 <delimiter> Item2 <delimiter> ...
Position ID Resetting: To correctly score Item2, the model must perceive it as immediately following the Query. It cannot know Item1 exists.
SGLang calculates a diff tensor to reset position IDs after every delimiter.
token_pos_in_items_ptr stores these relative positions.
Prefix Pointers: prefix_len_ptr tells the kernel where the shared Query ends and the distinct Items begin.
3. FlashInfer CUDA Integration
The critical optimization happens in the forward_extend method within flashinfer_backend.py.
1. Paged KV Cache Wrapper
SGLang uses BatchPrefillWithPagedKVCacheWrapper for scoring. This is vital because the Query KV cache is computed once and stored in the Radix Cache (paged memory). The Items reuse this cached prefix without recomputing it.
2. Kernel Invocation
When calling the FlashInfer CUDA wrapper, SGLang passes the specialized pointers instead of a generic attention mask:
Python
wrapper_paged.begin_forward(
    # ... standard pointers ...
    custom_mask=None, # Explicitly disabled for multi-item
    # ...
    prefix_len_ptr=prefix_len_ptr,
    token_pos_in_items_ptr=token_pos_in_items_ptr,
    token_pos_in_items_len=token_pos_in_items_len,
    max_item_len_ptr=max_item_len_ptr,
)
3. What the CUDA Kernel Does (Conceptually)
Under the hood, the FlashInfer CUDA kernel for this mode performs "Ragged Attention" with the following logic for a token $t$ in Item_i:
Keys/Values: It attends to the KV cache of the Query (shared prefix) AND the KV cache of the current Item_i up to token $t$.
Masking: It creates a causal mask that excludes Item_1...Item_{i-1}. To Item_i, it looks like those previous items essentially don't exist.
Position Embeddings (RoPE): It uses the re-calculated token_pos_in_items_ptr to generate Rotary Positional Embeddings that place Item_i immediately after the Query, preserving relative distance.
Summary for JAX Reimplementation
To replicate this behavior in JAX:
Input Packing: You will likely need to pack Query + Item1 + Item2 into a single dense array (with padding) or handling ragged arrays if using Pallas/Triton.
Block-Diagonal Masking: You cannot use a standard causal mask (lower triangular). You need a Block-Diagonal Mask where:
Query attends to Query (Causal).
Items attend to Query (Full).
Items attend to themselves (Causal).
Items DO NOT attend to other Items (Masked).
Position ID Manipulation: You must supply a custom position_ids array to your model layer that resets the count for every Item, ensuring the positional encodings reflect [0...Q_len, Q_len+1...Q_len+I_len] rather than a continuous sequence.
Implementation
To reimplement the SGLang /v1/score API in JAX, you need to modify the request handling pipeline to support "prefill-only" execution that returns log probabilities for specific target segments without triggering the decoding loop.
Here is the step-by-step breakdown referencing the specific files in the sgl-jax codebase.
Step 1: Update TokenizerManager to Handle Scoring
File: python/sgl_jax/srt/managers/tokenizer_manager.py (Inferred location based on usage in http_server.py)
You need to implement score_request to process the inputs (Query + Candidate Items) into a batch of requests.
Goal: Convert ScoringRequest into internal Req objects marked for scoring.
Changes:
Add a score_request method.
Tokenize the query.
Tokenize each item in request.items.
Concatenate [query_ids + item_ids] for each item.
Create Req objects. Crucially, set return_logprob=True and logprob_start_len=len(query_ids) (so we only score the item part).
Send these requests to the Scheduler.
Python
# Pseudo-code for TokenizerManager.score_request
async def score_request(self, query, items, ...):
    query_ids = self.tokenizer.encode(query)
    futures = []
    
    for item in items:
        item_ids = self.tokenizer.encode(item)
        full_ids = query_ids + item_ids
        
        # Create a request that demands logprobs starting from the item
        req = Req(
            rid=f"score-{uuid()}",
            input_ids=full_ids,
            return_logprob=True,
            logprob_start_len=len(query_ids),
            sampling_params=SamplingParams(max_new_tokens=0), # 0 tokens = prefill only
            ...
        )
        futures.append(self.send_to_scheduler(req))
        
    # Wait for all futures and assemble scores
    return await asyncio.gather(*futures)
Step 2: Modify Scheduler to Dispatch Scoring Requests
File: python/sgl_jax/srt/managers/scheduler.py
The scheduler needs to recognize that these requests should finish immediately after the prefill phase.
Goal: Process the scoring batch and ensure it does not transition to the decoding queue.
Changes:
Input Handling: In handle_generate_request (or a new handle_score_request if you add a new dispatcher type), ensure max_new_tokens=0 is respected.
Queue Logic: The existing logic might auto-abort requests with 0 output tokens. Verify validate_input_length doesn't reject them.
Batch Processing: In process_batch_result_prefill:
If batch.max_new_tokens == 0, mark the request as finished immediately after the prefill step.
Do not add it to self.running_batch (the decode queue).
Python
#
def process_batch_result_prefill(self, batch, result, ...):
    # ... existing logic ...
    
    for i, req in enumerate(batch.reqs):
        # NEW: Check if this was a scoring request (max_new_tokens=0)
        if req.sampling_params.max_new_tokens == 0:
            # Extract logprobs from result and finish request
            req.output_stats = extract_logprobs(result, i)
            req.finished = True
            self.send_to_detokenizer(req)
            continue 
            
        # ... existing decode transition logic ...
Step 3: Implement Logprob Extraction in ModelRunner
File: python/sgl_jax/srt/model_executor/model_runner.py
You need to ensure the model computes log probabilities for the entire prompt sequence (or the specific suffix) during the prefill forward pass.
Goal: Calculate log probabilities for input_ids when return_logprob is set.
Changes:
Logits Metadata: In _forward_raw, ensure logits_metadata is configured to return logits for the positions corresponding to items. By default, SGLang might only return the last token's logit for generation.
Forward Pass:
Python

#
def _forward_raw(self, forward_batch, logits_metadata):
    # ...
    # Ensure logits_metadata.return_logprob is handled for PREFILL mode
    # Standard generation only cares about the last token in prefill.
    # Scoring needs logits for the suffix tokens.


Compute Logprobs: Use the existing JIT-compiled compute_logprobs.
You may need to modify compute_logprobs to accept a sequence of target IDs rather than just a single next_token.
Step 4: Expose via HTTP Server
File: python/sgl_jax/srt/entrypoints/http_server.py
This file is already mostly set up, but you need to confirm OpenAIServingScore is initialized correctly with the updated Tokenizer Manager.
Changes:
Verify the endpoint /v1/score is registered (it appears to be in the provided file).
Ensure the OpenAIServingScore class (imported from openai.serving_score) correctly calls the new score_request method you added in Step 1.
Step 5: (Advanced) Block-Diagonal Masking (Optional Optimization)
Files: python/sgl_jax/srt/layers/attention/*.py
If performance is critical, you can implement the "Multi-Item" optimization mentioned in the GPU deep dive.
Concept: Instead of batching [Query+Item1], [Query+Item2], you batch [Query, Item1, Item2, ...] as a single sequence.
Change:
Modify the JAX attention mechanism (Pallas/Triton kernel) to accept a custom mask.
The mask should allow Item tokens to attend to Query tokens but block Item tokens from attending to each other.
This is significantly more complex in JAX/Flax than standard batching and should be attempting only after the basic implementation works.

