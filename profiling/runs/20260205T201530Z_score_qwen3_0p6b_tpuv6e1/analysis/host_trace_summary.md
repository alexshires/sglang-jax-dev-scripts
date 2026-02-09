# Host Trace Summary

Trace file: `t1v-n-3d21a6bb-w-0.trace.json.gz`

Time window: **385.047 ms**

Top events by total duration (ms):

| Event | Count | Total ms |
|---|---|---|
| $threading.py:637 wait | 2 | 12085.414 |
| $threading.py:323 wait | 2 | 12085.402 |
| $scheduler.py:645 recv_requests | 16949 | 296.140 |
| $socket.py:961 recv_pyobj | 33898 | 267.623 |
| $error.py:120 __init__ | 33898 | 105.145 |
| $error.py:45 __init__ | 33898 | 60.099 |
| $<frozen importlib._bootstrap>:1390 _handle_fromlist | 67796 | 47.503 |
| $scheduler.py:1025 check_memory | 16949 | 32.169 |
| $scheduler.py:1096 get_next_batch_to_run | 16949 | 19.034 |
| $scheduler.py:670 process_input_requests | 16949 | 18.199 |
| $utils.py:18 __call__ | 1 | 16.586 |
| $scheduler_profiler_mixing.py:128 profile | 1 | 16.584 |
| $scheduler_profiler_mixing.py:23 start_profile | 1 | 16.583 |
| $profiler.py:101 start_trace | 1 | 16.574 |
| $scheduler.py:1069 _get_token_info | 16949 | 13.408 |
| $sys _getframe | 169486 | 12.667 |
| $builtins hasattr | 101695 | 11.277 |
| $builtins isinstance | 135592 | 10.230 |
| $_zmq.py:183 zmq.backend.cython._zmq._check_rc | 33898 | 7.857 |
| $_zmq.py:1403 zmq.backend.cython._zmq._recv_copy | 33898 | 7.674 |
