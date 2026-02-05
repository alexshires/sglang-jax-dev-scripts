| Op type | Operation | Occurrences | Total us | Avg us | % of non-idle |
|---|---|---|---|---|---|
| pallas_call | jit(jitted_run_model)/QWen3Attention/FlashAttention/jit(ragged_paged_attention)/RPA-bq_32- | 1.0 | 453.051 | 453.051 | 29.90% |
| dot_general | jit(jitted_run_model)/LogitsProcessor/dot_general | 1.0 | 200.641 | 200.641 | 13.24% |
| Unknown | conditional.3 | 1.0 | 155.970 | 155.970 | 10.29% |
| reduce_sum | jit(jitted_sampler)/Sampler/cond/branch_0_fun/while/body/closed_call/reduce_sum | 31.0 | 102.006 | 3.291 | 6.73% |
| dot_general | jit(jitted_run_model)/Qwen3MLP/down_proj/dot_general | 1.0 | 71.736 | 71.736 | 4.73% |
| transpose | jit(jitted_run_model)/QWen3Attention/FlashAttention/jit(ragged_paged_attention)/transpose | 1.0 | 71.304 | 71.304 | 4.71% |
| Unknown | while.6 | 1.0 | 51.994 | 51.994 | 3.43% |
| Unknown | while.7 | 1.0 | 51.981 | 51.981 | 3.43% |
| dot_general | jit(jitted_run_model)/Qwen3MLP/gate_proj/dot_general | 1.0 | 41.394 | 41.394 | 2.73% |
| reduce | jit(jitted_sampler)/Sampler/cond/branch_0_fun/cond/branch_0_fun/reduce | 1.0 | 30.039 | 30.039 | 1.98% |
| dot_general | jit(jitted_run_model)/QWen3Attention/o_proj/dot_general | 1.0 | 28.376 | 28.376 | 1.87% |
| dot_general | jit(jitted_run_model)/QWen3Attention/q_proj/dot_general | 1.0 | 23.124 | 23.124 | 1.53% |
| dot_general | jit(jitted_run_model)/QWen3Attention/k_proj/dot_general | 1.0 | 17.279 | 17.279 | 1.14% |
| sub | jit(jitted_run_model)/QWen3Attention/RotaryEmbedding/sub | 1.0 | 15.852 | 15.852 | 1.05% |
| dot_general | jit(jitted_run_model)/QWen3Attention/v_proj/dot_general | 1.0 | 14.336 | 14.336 | 0.95% |
| reshape | jit(jitted_run_model)/QWen3Attention/FlashAttention/reshape | 1.0 | 14.168 | 14.168 | 0.94% |
| gather | jit(jitted_run_model)/LogitsProcessor/gather | 1.0 | 12.610 | 12.610 | 0.83% |
| Unknown | pad_maximum_fusion | 1.0 | 12.511 | 12.511 | 0.83% |
| reduce_sum | jit(jitted_run_model)/LogitsProcessor/jit(log_softmax)/reduce_sum | 1.0 | 9.117 | 9.117 | 0.60% |
| reduce_sum | jit(jitted_run_model)/QWen3Attention/q_norm/reduce_sum | 1.0 | 9.091 | 9.091 | 0.60% |
