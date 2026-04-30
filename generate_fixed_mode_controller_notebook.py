from __future__ import annotations

import pprint
from pathlib import Path

import nbformat as nbf


WORKLOADS = [
    "short_prompt_short_output",
    "short_prompt_long_output",
    "long_prompt_short_output",
    "long_prompt_long_output",
    "shared_prefix_chat_v0",
    "shared_prefix_chat_v1",
    "memory_pressure_long_context",
    "mmlu_pro_eval",
    "gsm8k_eval",
    "truthfulqa_eval",
    "gpqa_eval",
    "mlu_eval",
    "mt_bench_eval",
    "alpacaeval2_lc_eval",
]

THROUGHPUT_TPS = {
    "fp16_baseline": {
        "short_prompt_short_output": 45.6,
        "short_prompt_long_output": 46.3,
        "long_prompt_short_output": 41.5,
        "long_prompt_long_output": 45.5,
        "shared_prefix_chat_v0": 44.8,
        "shared_prefix_chat_v1": 44.5,
        "memory_pressure_long_context": 45.1,
        "mmlu_pro_eval": 18.9,
        "gsm8k_eval": 49.0,
        "truthfulqa_eval": 22.1,
        "gpqa_eval": 18.9,
        "mlu_eval": 19.7,
        "mt_bench_eval": 49.2,
        "alpacaeval2_lc_eval": 49.3,
    },
    "gptq_4bit": {
        "short_prompt_short_output": 116.8,
        "short_prompt_long_output": 136.3,
        "long_prompt_short_output": 91.7,
        "long_prompt_long_output": 123.9,
        "shared_prefix_chat_v0": 112.4,
        "shared_prefix_chat_v1": 111.1,
        "memory_pressure_long_context": 113.9,
        "mmlu_pro_eval": 16.6,
        "gsm8k_eval": 137.0,
        "truthfulqa_eval": 18.6,
        "gpqa_eval": 16.3,
        "mlu_eval": 17.0,
        "mt_bench_eval": 140.4,
        "alpacaeval2_lc_eval": 138.6,
    },
    "speculative_decoding": {
        "short_prompt_short_output": 100.0,
        "short_prompt_long_output": 98.2,
        "long_prompt_short_output": 75.6,
        "long_prompt_long_output": 107.0,
        "shared_prefix_chat_v0": 85.8,
        "shared_prefix_chat_v1": 85.6,
        "memory_pressure_long_context": 71.8,
        "mmlu_pro_eval": 12.7,
        "gsm8k_eval": 123.9,
        "truthfulqa_eval": 14.3,
        "gpqa_eval": 12.8,
        "mlu_eval": 13.2,
        "mt_bench_eval": 111.0,
        "alpacaeval2_lc_eval": 107.9,
    },
    "gptq_plus_prefix_caching": {
        "short_prompt_short_output": 127.9,
        "short_prompt_long_output": 139.5,
        "long_prompt_short_output": 122.1,
        "long_prompt_long_output": 134.1,
        "shared_prefix_chat_v0": 129.3,
        "shared_prefix_chat_v1": 130.2,
        "memory_pressure_long_context": 129.5,
    },
    "int8_plus_continuous_batching": {
        "short_prompt_short_output": 309.9,
        "short_prompt_long_output": 221.6,
        "long_prompt_short_output": 192.4,
        "long_prompt_long_output": 283.4,
        "shared_prefix_chat_v0": 251.8,
        "shared_prefix_chat_v1": 246.9,
        "memory_pressure_long_context": 244.3,
    },
    "prefix_caching": {
        "short_prompt_short_output": 65.0,
        "short_prompt_long_output": 67.7,
        "long_prompt_short_output": 63.0,
        "long_prompt_long_output": 66.7,
        "shared_prefix_chat_v0": 65.5,
        "shared_prefix_chat_v1": 65.6,
        "memory_pressure_long_context": 65.1,
        "mmlu_pro_eval": 17.0,
        "gsm8k_eval": 67.4,
        "truthfulqa_eval": 19.0,
        "gpqa_eval": 17.0,
        "mlu_eval": 17.7,
        "mt_bench_eval": 67.7,
        "alpacaeval2_lc_eval": 67.0,
    },
    "continuous_batching": {
        "short_prompt_short_output": 230.6,
        "short_prompt_long_output": 184.4,
        "long_prompt_short_output": 140.1,
        "long_prompt_long_output": 211.0,
        "shared_prefix_chat_v0": 185.4,
        "shared_prefix_chat_v1": 182.2,
        "memory_pressure_long_context": 181.9,
    },
}

LATENCY_MS = {
    "fp16_baseline": {
        "short_prompt_short_output": 702.0,
        "short_prompt_long_output": 2762.0,
        "long_prompt_short_output": 772.0,
        "long_prompt_long_output": 2814.0,
        "shared_prefix_chat_v0": 1428.0,
        "shared_prefix_chat_v1": 1439.0,
        "memory_pressure_long_context": 2839.0,
        "mmlu_pro_eval": 54.0,
        "gsm8k_eval": 3131.0,
        "truthfulqa_eval": 45.0,
        "gpqa_eval": 54.0,
        "mlu_eval": 52.0,
        "mt_bench_eval": 9571.0,
        "alpacaeval2_lc_eval": 7580.0,
    },
    "gptq_4bit": {
        "short_prompt_short_output": 283.0,
        "short_prompt_long_output": 939.0,
        "long_prompt_short_output": 349.0,
        "long_prompt_long_output": 1034.0,
        "shared_prefix_chat_v0": 569.0,
        "shared_prefix_chat_v1": 576.0,
        "memory_pressure_long_context": 1124.0,
        "mmlu_pro_eval": 63.0,
        "gsm8k_eval": 1086.0,
        "truthfulqa_eval": 54.0,
        "gpqa_eval": 62.0,
        "mlu_eval": 60.0,
        "mt_bench_eval": 3129.0,
        "alpacaeval2_lc_eval": 2512.0,
    },
    "speculative_decoding": {
        "short_prompt_short_output": 320.0,
        "short_prompt_long_output": 1304.0,
        "long_prompt_short_output": 423.0,
        "long_prompt_long_output": 1197.0,
        "shared_prefix_chat_v0": 746.0,
        "shared_prefix_chat_v1": 748.0,
        "memory_pressure_long_context": 1784.0,
        "mmlu_pro_eval": 80.0,
        "gsm8k_eval": 1242.0,
        "truthfulqa_eval": 70.0,
        "gpqa_eval": 79.0,
        "mlu_eval": 77.0,
        "mt_bench_eval": 4237.0,
        "alpacaeval2_lc_eval": 3443.0,
    },
    "gptq_plus_prefix_caching": {
        "short_prompt_short_output": 252.0,
        "short_prompt_long_output": 917.0,
        "long_prompt_short_output": 264.0,
        "long_prompt_long_output": 954.0,
        "shared_prefix_chat_v0": 496.0,
        "shared_prefix_chat_v1": 492.0,
        "memory_pressure_long_context": 990.0,
    },
    "int8_plus_continuous_batching": {
        "short_prompt_short_output": 413.0,
        "short_prompt_long_output": 1458.0,
        "long_prompt_short_output": 665.0,
        "long_prompt_long_output": 1807.0,
        "shared_prefix_chat_v0": 1017.0,
        "shared_prefix_chat_v1": 1037.0,
        "memory_pressure_long_context": 2096.0,
    },
    "prefix_caching": {
        "short_prompt_short_output": 492.0,
        "short_prompt_long_output": 1892.0,
        "long_prompt_short_output": 509.0,
        "long_prompt_long_output": 1919.0,
        "shared_prefix_chat_v0": 977.0,
        "shared_prefix_chat_v1": 976.0,
        "memory_pressure_long_context": 1967.0,
        "mmlu_pro_eval": 60.0,
        "gsm8k_eval": 2251.0,
        "truthfulqa_eval": 53.0,
        "gpqa_eval": 60.0,
        "mlu_eval": 58.0,
        "mt_bench_eval": 6997.0,
        "alpacaeval2_lc_eval": 5561.0,
    },
    "continuous_batching": {
        "short_prompt_short_output": 555.0,
        "short_prompt_long_output": 1979.0,
        "long_prompt_short_output": 914.0,
        "long_prompt_long_output": 2426.0,
        "shared_prefix_chat_v0": 1381.0,
        "shared_prefix_chat_v1": 1405.0,
        "memory_pressure_long_context": 2815.0,
    },
}

ENERGY_J_PER_TOKEN = {
    "fp16_baseline": {
        "short_prompt_short_output": 3.91,
        "short_prompt_long_output": 3.88,
        "long_prompt_short_output": 5.05,
        "long_prompt_long_output": 4.21,
        "shared_prefix_chat_v0": 4.67,
        "shared_prefix_chat_v1": 4.71,
        "memory_pressure_long_context": 4.41,
        "mmlu_pro_eval": 8.22,
        "gsm8k_eval": 3.82,
        "truthfulqa_eval": 8.38,
        "gpqa_eval": 9.88,
        "mlu_eval": 7.95,
        "mt_bench_eval": 3.85,
        "alpacaeval2_lc_eval": 3.82,
    },
    "gptq_4bit": {
        "short_prompt_short_output": 1.81,
        "short_prompt_long_output": 1.50,
        "long_prompt_short_output": 2.73,
        "long_prompt_long_output": 1.86,
        "shared_prefix_chat_v0": 2.37,
        "shared_prefix_chat_v1": 2.42,
        "memory_pressure_long_context": 2.06,
        "mmlu_pro_eval": 7.53,
        "gsm8k_eval": 1.48,
        "truthfulqa_eval": 9.77,
        "gpqa_eval": 9.87,
        "mlu_eval": 8.29,
        "mt_bench_eval": 1.48,
        "alpacaeval2_lc_eval": 1.53,
    },
    "speculative_decoding": {
        "short_prompt_short_output": 1.88,
        "short_prompt_long_output": 2.02,
        "long_prompt_short_output": 3.12,
        "long_prompt_long_output": 2.01,
        "shared_prefix_chat_v0": 2.91,
        "shared_prefix_chat_v1": 2.95,
        "memory_pressure_long_context": 3.08,
        "mmlu_pro_eval": 7.69,
        "gsm8k_eval": 1.60,
        "truthfulqa_eval": 8.79,
        "gpqa_eval": 9.24,
        "mlu_eval": 7.94,
        "mt_bench_eval": 1.88,
        "alpacaeval2_lc_eval": 1.92,
    },
    "gptq_plus_prefix_caching": {
        "short_prompt_short_output": 1.18,
        "short_prompt_long_output": 1.40,
        "long_prompt_short_output": 1.43,
        "long_prompt_long_output": 1.42,
        "shared_prefix_chat_v0": 1.40,
        "shared_prefix_chat_v1": 1.35,
        "memory_pressure_long_context": 1.45,
    },
    "int8_plus_continuous_batching": {
        "short_prompt_short_output": 0.62,
        "short_prompt_long_output": 0.85,
        "long_prompt_short_output": 1.37,
        "long_prompt_long_output": 0.77,
        "shared_prefix_chat_v0": 1.04,
        "shared_prefix_chat_v1": 1.05,
        "memory_pressure_long_context": 0.98,
    },
    "prefix_caching": {
        "short_prompt_short_output": 3.13,
        "short_prompt_long_output": 3.38,
        "long_prompt_short_output": 3.49,
        "long_prompt_long_output": 3.47,
        "shared_prefix_chat_v0": 3.43,
        "shared_prefix_chat_v1": 3.44,
        "memory_pressure_long_context": 3.50,
        "mmlu_pro_eval": 7.40,
        "gsm8k_eval": 3.43,
        "truthfulqa_eval": 9.07,
        "gpqa_eval": 9.48,
        "mlu_eval": 8.07,
        "mt_bench_eval": 3.46,
        "alpacaeval2_lc_eval": 3.46,
    },
    "continuous_batching": {
        "short_prompt_short_output": 1.05,
        "short_prompt_long_output": 1.28,
        "long_prompt_short_output": 2.13,
        "long_prompt_long_output": 1.26,
        "shared_prefix_chat_v0": 1.54,
        "shared_prefix_chat_v1": 1.61,
        "memory_pressure_long_context": 1.54,
    },
}

PEAK_GPU_MEMORY_MB = {
    "fp16_baseline": {
        "short_prompt_short_output": 32673.0,
        "short_prompt_long_output": 32676.0,
        "long_prompt_short_output": 32812.0,
        "long_prompt_long_output": 32812.0,
        "shared_prefix_chat_v0": 32809.0,
        "shared_prefix_chat_v1": 32806.0,
        "memory_pressure_long_context": 36817.0,
        "mmlu_pro_eval": 32686.0,
        "gsm8k_eval": 32670.0,
        "truthfulqa_eval": 32672.0,
        "gpqa_eval": 32687.0,
        "mlu_eval": 32682.0,
        "mt_bench_eval": 32669.0,
        "alpacaeval2_lc_eval": 32661.0,
    },
    "gptq_4bit": {
        "short_prompt_short_output": 32978.0,
        "short_prompt_long_output": 32978.0,
        "long_prompt_short_output": 33078.0,
        "long_prompt_long_output": 33078.0,
        "shared_prefix_chat_v0": 33075.0,
        "shared_prefix_chat_v1": 33073.0,
        "memory_pressure_long_context": 37013.0,
        "mmlu_pro_eval": 32986.0,
        "gsm8k_eval": 32974.0,
        "truthfulqa_eval": 32975.0,
        "gpqa_eval": 32986.0,
        "mlu_eval": 32983.0,
        "mt_bench_eval": 32974.0,
        "alpacaeval2_lc_eval": 32968.0,
    },
    "speculative_decoding": {
        "short_prompt_short_output": 32861.0,
        "short_prompt_long_output": 32861.0,
        "long_prompt_short_output": 32998.0,
        "long_prompt_long_output": 32998.0,
        "shared_prefix_chat_v0": 32995.0,
        "shared_prefix_chat_v1": 32992.0,
        "memory_pressure_long_context": 36910.0,
        "mmlu_pro_eval": 32872.0,
        "gsm8k_eval": 32861.0,
        "truthfulqa_eval": 32861.0,
        "gpqa_eval": 32873.0,
        "mlu_eval": 32870.0,
        "mt_bench_eval": 32863.0,
        "alpacaeval2_lc_eval": 32861.0,
    },
    "gptq_plus_prefix_caching": {
        "short_prompt_short_output": 33006.0,
        "short_prompt_long_output": 33005.0,
        "long_prompt_short_output": 33010.0,
        "long_prompt_long_output": 33005.0,
        "shared_prefix_chat_v0": 33011.0,
        "shared_prefix_chat_v1": 33010.0,
        "memory_pressure_long_context": 36845.0,
    },
    "int8_plus_continuous_batching": {
        "short_prompt_short_output": 32965.0,
        "short_prompt_long_output": 32965.0,
        "long_prompt_short_output": 33187.0,
        "long_prompt_long_output": 33187.0,
        "shared_prefix_chat_v0": 33181.0,
        "shared_prefix_chat_v1": 33172.0,
        "memory_pressure_long_context": 36925.0,
    },
    "prefix_caching": {
        "short_prompt_short_output": 32937.0,
        "short_prompt_long_output": 32936.0,
        "long_prompt_short_output": 32943.0,
        "long_prompt_long_output": 32936.0,
        "shared_prefix_chat_v0": 32943.0,
        "shared_prefix_chat_v1": 32943.0,
        "memory_pressure_long_context": 36703.0,
        "mmlu_pro_eval": 32961.0,
        "gsm8k_eval": 32946.0,
        "truthfulqa_eval": 32947.0,
        "gpqa_eval": 32962.0,
        "mlu_eval": 32957.0,
        "mt_bench_eval": 32946.0,
        "alpacaeval2_lc_eval": 32938.0,
    },
    "continuous_batching": {
        "short_prompt_short_output": 33038.0,
        "short_prompt_long_output": 33038.0,
        "long_prompt_short_output": 33262.0,
        "long_prompt_long_output": 33262.0,
        "shared_prefix_chat_v0": 33256.0,
        "shared_prefix_chat_v1": 33247.0,
        "memory_pressure_long_context": 36963.0,
    },
}

PREFILL_SHARE_PCT = {
    "fp16_baseline": {
        "short_prompt_short_output": 3.7,
        "short_prompt_long_output": 0.9,
        "long_prompt_short_output": 14.2,
        "long_prompt_long_output": 3.9,
        "shared_prefix_chat_v0": 7.2,
        "shared_prefix_chat_v1": 7.4,
        "memory_pressure_long_context": 6.1,
        "mmlu_pro_eval": 60.2,
        "gsm8k_eval": 0.8,
        "truthfulqa_eval": 54.9,
        "gpqa_eval": 61.4,
        "mlu_eval": 59.9,
        "mt_bench_eval": 0.3,
        "alpacaeval2_lc_eval": 0.3,
    },
    "gptq_4bit": {
        "short_prompt_short_output": 17.4,
        "short_prompt_long_output": 5.2,
        "long_prompt_short_output": 36.3,
        "long_prompt_long_output": 12.2,
        "shared_prefix_chat_v0": 20.9,
        "shared_prefix_chat_v1": 21.9,
        "memory_pressure_long_context": 17.1,
        "mmlu_pro_eval": 88.0,
        "gsm8k_eval": 4.3,
        "truthfulqa_eval": 86.8,
        "gpqa_eval": 88.6,
        "mlu_eval": 88.1,
        "mt_bench_eval": 1.5,
        "alpacaeval2_lc_eval": 1.8,
    },
    "speculative_decoding": {
        "short_prompt_short_output": 11.6,
        "short_prompt_long_output": 2.8,
        "long_prompt_short_output": 30.7,
        "long_prompt_long_output": 10.9,
        "shared_prefix_chat_v0": 16.4,
        "shared_prefix_chat_v1": 17.0,
        "memory_pressure_long_context": 11.5,
        "mmlu_pro_eval": 55.2,
        "gsm8k_eval": 2.9,
        "truthfulqa_eval": 49.9,
        "gpqa_eval": 55.5,
        "mlu_eval": 54.0,
        "mt_bench_eval": 0.8,
        "alpacaeval2_lc_eval": 1.0,
    },
    "gptq_plus_prefix_caching": {
        "short_prompt_short_output": 13.8,
        "short_prompt_long_output": 3.2,
        "long_prompt_short_output": 14.8,
        "long_prompt_long_output": 3.5,
        "shared_prefix_chat_v0": 7.6,
        "shared_prefix_chat_v1": 7.5,
        "memory_pressure_long_context": 4.6,
    },
    "int8_plus_continuous_batching": {
        "short_prompt_short_output": 15.1,
        "short_prompt_long_output": 4.3,
        "long_prompt_short_output": 43.1,
        "long_prompt_long_output": 15.9,
        "shared_prefix_chat_v0": 25.8,
        "shared_prefix_chat_v1": 26.3,
        "memory_pressure_long_context": 13.1,
    },
    "prefix_caching": {
        "short_prompt_short_output": 7.7,
        "short_prompt_long_output": 2.0,
        "long_prompt_short_output": 9.0,
        "long_prompt_long_output": 2.2,
        "shared_prefix_chat_v0": 4.6,
        "shared_prefix_chat_v1": 4.6,
        "memory_pressure_long_context": 2.7,
        "mmlu_pro_eval": 75.1,
        "gsm8k_eval": 1.7,
        "truthfulqa_eval": 72.2,
        "gpqa_eval": 75.4,
        "mlu_eval": 74.6,
        "mt_bench_eval": 0.5,
        "alpacaeval2_lc_eval": 0.7,
    },
    "continuous_batching": {
        "short_prompt_short_output": 15.4,
        "short_prompt_long_output": 4.3,
        "long_prompt_short_output": 45.3,
        "long_prompt_long_output": 17.1,
        "shared_prefix_chat_v0": 27.6,
        "shared_prefix_chat_v1": 28.8,
        "memory_pressure_long_context": 13.8,
    },
}

DECODE_SHARE_PCT = {
    mode: {workload: round(100.0 - share, 1) for workload, share in values.items()}
    for mode, values in PREFILL_SHARE_PCT.items()
}

AUTOMATIC_ACCURACY_PCT = {
    "fp16_baseline": {
        "mmlu_pro_eval": 35.2,
        "gsm8k_eval": 83.2,
        "truthfulqa_eval": 57.4,
        "gpqa_eval": 27.4,
        "mlu_eval": 46.0,
    },
    "gptq_4bit": {
        "mmlu_pro_eval": 33.4,
        "gsm8k_eval": 78.4,
        "truthfulqa_eval": 54.6,
        "gpqa_eval": 29.6,
        "mlu_eval": 43.0,
    },
    "int8_quant": {
        "mmlu_pro_eval": 36.2,
        "gsm8k_eval": 83.8,
        "truthfulqa_eval": 56.6,
        "gpqa_eval": 28.0,
        "mlu_eval": 46.8,
    },
    "speculative_decoding": {
        "mmlu_pro_eval": 35.2,
        "gsm8k_eval": 83.2,
        "truthfulqa_eval": 57.4,
        "gpqa_eval": 27.4,
        "mlu_eval": 46.0,
    },
    "prefix_caching": {
        "mmlu_pro_eval": 35.2,
        "gsm8k_eval": 83.6,
        "truthfulqa_eval": 57.4,
        "gpqa_eval": 27.6,
        "mlu_eval": 46.0,
    },
}

SUPPLEMENT_METRIC_ROWS = [
    {
        "mode_name": "int8_quant",
        "workload_name": "mmlu_pro_eval",
        "total_latency_ms": 51.551425,
        "tokens_per_second": 19.811163,
        "energy_per_token_j": 6.351850,
        "peak_gpu_memory_mb": 32982.493262,
        "source": "final_test_notebook_extract",
    },
    {
        "mode_name": "int8_quant",
        "workload_name": "gsm8k_eval",
        "total_latency_ms": 1600.159060,
        "tokens_per_second": 90.751498,
        "energy_per_token_j": 2.126914,
        "peak_gpu_memory_mb": 32963.570312,
        "source": "final_test_notebook_extract",
    },
    {
        "mode_name": "int8_quant",
        "workload_name": "truthfulqa_eval",
        "total_latency_ms": 45.876493,
        "tokens_per_second": 21.798283,
        "energy_per_token_j": 8.404530,
        "peak_gpu_memory_mb": 32966.135742,
        "source": "final_test_notebook_extract",
    },
]

WORKLOAD_SIGNALS = [
    {
        "workload_name": "short_prompt_short_output",
        "prompt_tokens": 128,
        "expected_output_tokens": 32,
        "shared_prefix": False,
        "memory_pressure": False,
        "batch_pressure": "normal",
        "quality_budget": "relaxed",
        "task_family": "synthetic_interactive",
    },
    {
        "workload_name": "short_prompt_long_output",
        "prompt_tokens": 128,
        "expected_output_tokens": 128,
        "shared_prefix": False,
        "memory_pressure": False,
        "batch_pressure": "normal",
        "quality_budget": "relaxed",
        "task_family": "synthetic_generation",
    },
    {
        "workload_name": "long_prompt_short_output",
        "prompt_tokens": 1024,
        "expected_output_tokens": 32,
        "shared_prefix": False,
        "memory_pressure": False,
        "batch_pressure": "normal",
        "quality_budget": "relaxed",
        "task_family": "synthetic_interactive",
    },
    {
        "workload_name": "long_prompt_long_output",
        "prompt_tokens": 1024,
        "expected_output_tokens": 128,
        "shared_prefix": False,
        "memory_pressure": False,
        "batch_pressure": "normal",
        "quality_budget": "relaxed",
        "task_family": "synthetic_generation",
    },
    {
        "workload_name": "shared_prefix_chat_v0",
        "prompt_tokens": 1024,
        "expected_output_tokens": 64,
        "shared_prefix": True,
        "memory_pressure": False,
        "batch_pressure": "normal",
        "quality_budget": "medium",
        "task_family": "chat_shared_prefix",
    },
    {
        "workload_name": "shared_prefix_chat_v1",
        "prompt_tokens": 1024,
        "expected_output_tokens": 64,
        "shared_prefix": True,
        "memory_pressure": False,
        "batch_pressure": "normal",
        "quality_budget": "medium",
        "task_family": "chat_shared_prefix",
    },
    {
        "workload_name": "memory_pressure_long_context",
        "prompt_tokens": 2048,
        "expected_output_tokens": 128,
        "shared_prefix": False,
        "memory_pressure": True,
        "batch_pressure": "normal",
        "quality_budget": "medium",
        "task_family": "synthetic_long_context",
    },
    {
        "workload_name": "mmlu_pro_eval",
        "prompt_tokens": 512,
        "expected_output_tokens": 8,
        "shared_prefix": False,
        "memory_pressure": False,
        "batch_pressure": "normal",
        "quality_budget": "strict",
        "task_family": "mcq_reasoning",
    },
    {
        "workload_name": "gsm8k_eval",
        "prompt_tokens": 512,
        "expected_output_tokens": 384,
        "shared_prefix": False,
        "memory_pressure": False,
        "batch_pressure": "normal",
        "quality_budget": "strict",
        "task_family": "math_generation",
    },
    {
        "workload_name": "truthfulqa_eval",
        "prompt_tokens": 512,
        "expected_output_tokens": 8,
        "shared_prefix": False,
        "memory_pressure": False,
        "batch_pressure": "normal",
        "quality_budget": "strict",
        "task_family": "mcq_truthfulness",
    },
    {
        "workload_name": "gpqa_eval",
        "prompt_tokens": 512,
        "expected_output_tokens": 8,
        "shared_prefix": False,
        "memory_pressure": False,
        "batch_pressure": "normal",
        "quality_budget": "ultra_strict",
        "task_family": "mcq_science",
    },
    {
        "workload_name": "mlu_eval",
        "prompt_tokens": 512,
        "expected_output_tokens": 8,
        "shared_prefix": False,
        "memory_pressure": False,
        "batch_pressure": "normal",
        "quality_budget": "ultra_strict",
        "task_family": "mcq_multilingual",
    },
    {
        "workload_name": "mt_bench_eval",
        "prompt_tokens": 512,
        "expected_output_tokens": 1024,
        "shared_prefix": False,
        "memory_pressure": False,
        "batch_pressure": "normal",
        "quality_budget": "medium",
        "task_family": "judge_chat_quality",
    },
    {
        "workload_name": "alpacaeval2_lc_eval",
        "prompt_tokens": 512,
        "expected_output_tokens": 512,
        "shared_prefix": False,
        "memory_pressure": False,
        "batch_pressure": "normal",
        "quality_budget": "medium",
        "task_family": "judge_instruction_following",
    },
]

MODE_PROFILES = [
    {
        "mode_name": "fp16_baseline",
        "mode_role": "prefill-heavy safe branch",
        "accuracy_risk": "lowest",
        "notes": "Conversation-aligned conservative branch for prefill-heavy short-output requests.",
    },
    {
        "mode_name": "prefix_caching",
        "mode_role": "lossless default",
        "accuracy_risk": "lowest",
        "notes": "Default conservative branch when no stronger regime signal is present.",
    },
    {
        "mode_name": "gptq_4bit",
        "mode_role": "decode-heavy branch",
        "accuracy_risk": "medium-high",
        "notes": "Best single-mode decode win from the sweep for non-shared long-generation requests.",
    },
    {
        "mode_name": "gptq_plus_prefix_caching",
        "mode_role": "shared-prefix chat branch",
        "accuracy_risk": "medium-high",
        "notes": "Best measured chat-style shared-prefix combo in the sweep.",
    },
    {
        "mode_name": "int8_plus_continuous_batching",
        "mode_role": "high-batch serving extension",
        "accuracy_risk": "unknown",
        "notes": "Best throughput / energy under load; use only when queue depth or batch pressure is high.",
    },
    {
        "mode_name": "continuous_batching",
        "mode_role": "throughput baseline",
        "accuracy_risk": "unknown",
        "notes": "Useful batching baseline, but dominated by the INT8 batching hybrid in the measured serving case.",
    },
    {
        "mode_name": "int8_quant",
        "mode_role": "benchmark-safe alternate",
        "accuracy_risk": "low",
        "notes": "Kept in the benchmark-safe comparison set, but not the default controller branch in the updated conversation direction.",
    },
    {
        "mode_name": "speculative_decoding",
        "mode_role": "alternate decode branch",
        "accuracy_risk": "low",
        "notes": "Still useful as a comparison mode, but the new controller tree does not make it the default decode branch.",
    },
]


def lit(obj: object) -> str:
    return pprint.pformat(obj, width=100, sort_dicts=False)


IMPORTS_CODE = """
from __future__ import annotations

import json
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
from controller import classify_request, route_request
from controller.features import RequestFeatures

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except Exception:
    plt = None
    PLOTTING_AVAILABLE = False

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
""".strip()

DATA_CODE = f"""
WORKLOADS = {lit(WORKLOADS)}
THROUGHPUT_TPS = {lit(THROUGHPUT_TPS)}
LATENCY_MS = {lit(LATENCY_MS)}
ENERGY_J_PER_TOKEN = {lit(ENERGY_J_PER_TOKEN)}
PEAK_GPU_MEMORY_MB = {lit(PEAK_GPU_MEMORY_MB)}
PREFILL_SHARE_PCT = {lit(PREFILL_SHARE_PCT)}
DECODE_SHARE_PCT = {lit(DECODE_SHARE_PCT)}
AUTOMATIC_ACCURACY_PCT = {lit(AUTOMATIC_ACCURACY_PCT)}
SUPPLEMENT_METRIC_ROWS = {lit(SUPPLEMENT_METRIC_ROWS)}
WORKLOAD_SIGNALS = {lit(WORKLOAD_SIGNALS)}
MODE_PROFILES = {lit(MODE_PROFILES)}
""".strip()

HELPERS_CODE = """
def nested_metric_to_long(metric_name: str, nested: dict[str, dict[str, float]], source: str) -> pd.DataFrame:
    rows = []
    for mode_name, workload_map in nested.items():
        for workload_name, value in workload_map.items():
            rows.append(
                {
                    "mode_name": mode_name,
                    "workload_name": workload_name,
                    metric_name: value,
                    "source": source,
                }
            )
    return pd.DataFrame(rows)


def build_metric_table() -> pd.DataFrame:
    parts = []
    for metric_name, nested in [
        ("tokens_per_second", THROUGHPUT_TPS),
        ("total_latency_ms", LATENCY_MS),
        ("energy_per_token_j", ENERGY_J_PER_TOKEN),
        ("peak_gpu_memory_mb", PEAK_GPU_MEMORY_MB),
        ("prefill_share_pct", PREFILL_SHARE_PCT),
        ("decode_share_pct", DECODE_SHARE_PCT),
    ]:
        parts.append(nested_metric_to_long(metric_name, nested, source="user_curated"))

    merged = None
    for df in parts:
        metric_cols = [c for c in df.columns if c not in {"mode_name", "workload_name", "source"}]
        if merged is None:
            merged = df.copy()
        else:
            merged = merged.merge(
                df[["mode_name", "workload_name", *metric_cols]],
                on=["mode_name", "workload_name"],
                how="outer",
            )

    supplement_df = pd.DataFrame(SUPPLEMENT_METRIC_ROWS)
    if not supplement_df.empty:
        merged = merged.merge(
            supplement_df,
            on=["mode_name", "workload_name"],
            how="outer",
            suffixes=("", "_supplement"),
        )
        for metric_name in [
            "tokens_per_second",
            "total_latency_ms",
            "energy_per_token_j",
            "peak_gpu_memory_mb",
        ]:
            supplement_col = f"{metric_name}_supplement"
            if supplement_col in merged.columns:
                merged[metric_name] = merged[metric_name].fillna(merged[supplement_col])
                merged = merged.drop(columns=[supplement_col])
        merged["source"] = merged["source"].fillna(merged.get("source_supplement"))
        if "source_supplement" in merged.columns:
            merged = merged.drop(columns=["source_supplement"])

    return merged.sort_values(["workload_name", "mode_name"]).reset_index(drop=True)


def build_accuracy_table() -> pd.DataFrame:
    rows = []
    for mode_name, workload_map in AUTOMATIC_ACCURACY_PCT.items():
        for workload_name, value in workload_map.items():
            rows.append(
                {
                    "mode_name": mode_name,
                    "workload_name": workload_name,
                    "automatic_accuracy_pct": value,
                    "source": "user_curated",
                }
            )
    return pd.DataFrame(rows).sort_values(["workload_name", "mode_name"]).reset_index(drop=True)


metric_df = build_metric_table()
accuracy_df = build_accuracy_table()
workload_df = pd.DataFrame(WORKLOAD_SIGNALS)
mode_profile_df = pd.DataFrame(MODE_PROFILES)

baseline_phase = (
    metric_df[metric_df["mode_name"] == "fp16_baseline"]
    [["workload_name", "prefill_share_pct", "decode_share_pct"]]
    .rename(
        columns={
            "prefill_share_pct": "baseline_prefill_share_pct",
            "decode_share_pct": "baseline_decode_share_pct",
        }
    )
)

workload_df = workload_df.merge(baseline_phase, on="workload_name", how="left")
workload_df["phase_label"] = np.where(
    workload_df["baseline_prefill_share_pct"] >= 50.0,
    "prefill-heavy",
    "decode-heavy",
)
workload_df["output_to_prompt_ratio"] = workload_df["expected_output_tokens"] / workload_df["prompt_tokens"]

print("Metric rows:", len(metric_df))
print("Accuracy rows:", len(accuracy_df))
metric_df.head()
""".strip()

EXTRACT_CODE = """
def extract_supporting_tables_from_final_test(path: str | Path = "final_test (7) (2).ipynb"):
    path = Path(path)
    if not path.exists():
        return None

    notebook = json.loads(path.read_text(encoding="utf-8"))

    def collect_output_text(cell: dict) -> str:
        texts = []
        for out in cell.get("outputs", []):
            if "text" in out:
                val = out["text"]
                texts.append("".join(val) if isinstance(val, list) else str(val))
            data = out.get("data", {})
            for key in ("text/plain", "text/html"):
                if key in data:
                    val = data[key]
                    texts.append("".join(val) if isinstance(val, list) else str(val))
        return "\\n".join(texts)

    extracted = {
        "standard_dense_modes": None,
        "benchmark_safe_dense_modes": None,
        "automatic_accuracy_table": None,
        "best_by_latency_table": None,
        "best_by_energy_table": None,
    }

    cell4_blob = collect_output_text(notebook["cells"][4]) if len(notebook["cells"]) > 4 else ""
    if "Standard dense modes:" in cell4_blob:
        std_block = cell4_blob.split("Standard dense modes:", 1)[1].split("Benchmark-safe dense modes:", 1)[0]
        extracted["standard_dense_modes"] = [
            line.replace("-", "").strip() for line in std_block.splitlines() if line.strip().startswith("-")
        ]
    if "Benchmark-safe dense modes:" in cell4_blob:
        bench_block = cell4_blob.split("Benchmark-safe dense modes:", 1)[1].split("Total expanded workloads:", 1)[0]
        extracted["benchmark_safe_dense_modes"] = [
            line.replace("-", "").strip() for line in bench_block.splitlines() if line.strip().startswith("-")
        ]

    if len(notebook["cells"]) > 11:
        html_chunks = []
        for out in notebook["cells"][11].get("outputs", []):
            html = out.get("data", {}).get("text/html")
            if html is not None:
                html_chunks.append("".join(html) if isinstance(html, list) else str(html))
        for html in html_chunks:
            try:
                tables = pd.read_html(StringIO(html))
            except Exception:
                continue
            for df in tables:
                if {"benchmark", "mode_name", "total_latency_ms_mean"}.issubset(df.columns):
                    extracted["automatic_accuracy_table"] = df.copy()
                    break
            if extracted["automatic_accuracy_table"] is not None:
                break

    if len(notebook["cells"]) > 16:
        html_chunks = []
        for out in notebook["cells"][16].get("outputs", []):
            html = out.get("data", {}).get("text/html")
            if html is not None:
                html_chunks.append("".join(html) if isinstance(html, list) else str(html))
        found_tables = []
        for html in html_chunks:
            try:
                found_tables.extend(pd.read_html(StringIO(html)))
            except Exception:
                continue
        ranked = [df for df in found_tables if {"workload_name", "mode_name"}.issubset(df.columns)]
        if ranked:
            extracted["best_by_latency_table"] = ranked[0].copy()
        if len(ranked) > 1:
            extracted["best_by_energy_table"] = ranked[1].copy()

    return extracted


supporting_tables = extract_supporting_tables_from_final_test()
if supporting_tables is None:
    print("No local final_test notebook found; using the curated tables only.")
else:
    print("Standard dense modes extracted from final_test:", supporting_tables["standard_dense_modes"])
    print("Benchmark-safe dense modes extracted from final_test:", supporting_tables["benchmark_safe_dense_modes"])
    if supporting_tables["automatic_accuracy_table"] is not None:
        print("\\nNotebook-derived automatic accuracy / benchmark-safe rows:")
        display_cols = [
            "benchmark",
            "mode_name",
            "total_latency_ms_mean",
            "tokens_per_second_mean",
            "energy_per_token_j_mean",
            "metric_display_value",
        ]
        display(supporting_tables["automatic_accuracy_table"][display_cols])
    if supporting_tables["best_by_latency_table"] is not None:
        print("\\nNotebook-derived best-by-latency summary:")
        display(
            supporting_tables["best_by_latency_table"][
                ["workload_name", "mode_name", "latency_speedup_vs_baseline", "energy_ratio_vs_baseline"]
            ]
        )
    if supporting_tables["best_by_energy_table"] is not None:
        print("\\nNotebook-derived best-by-energy summary:")
        display(
            supporting_tables["best_by_energy_table"][
                ["workload_name", "mode_name", "energy_ratio_vs_baseline", "latency_speedup_vs_baseline"]
            ]
        )
""".strip()

ROUTE_CODE = """
routed = workload_df.copy()
routed[[
    "controller_phase_label",
    "controller_estimated_prefill_share_pct",
    "selected_mode",
    "route_reason",
]] = routed.apply(
    lambda row: pd.Series(
        (
            lambda features, classification, decision: (
                classification.label,
                classification.estimated_prefill_share_pct,
                decision.selected_mode_name,
                decision.reason,
            )
        )(
            RequestFeatures(
                prompt_tokens=int(row["prompt_tokens"]),
                expected_output_tokens=int(row["expected_output_tokens"]),
                shared_prefix=bool(row["shared_prefix"]),
                batch_pressure=str(row["batch_pressure"]),
                memory_pressure=bool(row["memory_pressure"]),
                workload_tag=str(row["task_family"]),
            ),
            classify_request(
                RequestFeatures(
                    prompt_tokens=int(row["prompt_tokens"]),
                    expected_output_tokens=int(row["expected_output_tokens"]),
                    shared_prefix=bool(row["shared_prefix"]),
                    batch_pressure=str(row["batch_pressure"]),
                    memory_pressure=bool(row["memory_pressure"]),
                    workload_tag=str(row["task_family"]),
                )
            ),
            route_request(
                RequestFeatures(
                    prompt_tokens=int(row["prompt_tokens"]),
                    expected_output_tokens=int(row["expected_output_tokens"]),
                    shared_prefix=bool(row["shared_prefix"]),
                    batch_pressure=str(row["batch_pressure"]),
                    memory_pressure=bool(row["memory_pressure"]),
                    workload_tag=str(row["task_family"]),
                )
            ),
        )
    ),
    axis=1,
)
routed = routed.merge(
    metric_df,
    left_on=["selected_mode", "workload_name"],
    right_on=["mode_name", "workload_name"],
    how="left",
).drop(columns=["mode_name", "source"])

baseline_metrics = (
    metric_df[metric_df["mode_name"] == "fp16_baseline"]
    [["workload_name", "total_latency_ms", "tokens_per_second", "energy_per_token_j", "peak_gpu_memory_mb"]]
    .rename(
        columns={
            "total_latency_ms": "baseline_total_latency_ms",
            "tokens_per_second": "baseline_tokens_per_second",
            "energy_per_token_j": "baseline_energy_per_token_j",
            "peak_gpu_memory_mb": "baseline_peak_gpu_memory_mb",
        }
    )
)
routed = routed.merge(baseline_metrics, on="workload_name", how="left")
routed["latency_speedup_vs_fp16"] = routed["baseline_total_latency_ms"] / routed["total_latency_ms"]
routed["energy_ratio_vs_fp16"] = routed["energy_per_token_j"] / routed["baseline_energy_per_token_j"]
routed["throughput_ratio_vs_fp16"] = routed["tokens_per_second"] / routed["baseline_tokens_per_second"]

selected_accuracy = accuracy_df.rename(
    columns={"mode_name": "selected_mode", "automatic_accuracy_pct": "selected_accuracy_pct"}
)
routed = routed.merge(
    selected_accuracy[["selected_mode", "workload_name", "selected_accuracy_pct"]],
    on=["selected_mode", "workload_name"],
    how="left",
)

baseline_accuracy = (
    accuracy_df[accuracy_df["mode_name"] == "fp16_baseline"][["workload_name", "automatic_accuracy_pct"]]
    .rename(columns={"automatic_accuracy_pct": "baseline_accuracy_pct"})
)
routed = routed.merge(baseline_accuracy, on="workload_name", how="left")
routed["accuracy_delta_pct"] = routed["selected_accuracy_pct"] - routed["baseline_accuracy_pct"]

selected_columns = [
    "workload_name",
    "task_family",
    "phase_label",
    "controller_phase_label",
    "controller_estimated_prefill_share_pct",
    "shared_prefix",
    "memory_pressure",
    "batch_pressure",
    "selected_mode",
    "total_latency_ms",
    "latency_speedup_vs_fp16",
    "energy_per_token_j",
    "energy_ratio_vs_fp16",
    "tokens_per_second",
    "throughput_ratio_vs_fp16",
    "selected_accuracy_pct",
    "baseline_accuracy_pct",
    "accuracy_delta_pct",
    "route_reason",
]

routed[selected_columns].sort_values(["workload_name"]).reset_index(drop=True)
""".strip()

SUMMARY_CODE = """
measured_routed = routed.copy()

summary_rows = []
for scope_name, scope_df in {
    "all_measured_workloads": measured_routed,
    "eval_workloads_with_accuracy": measured_routed[measured_routed["selected_accuracy_pct"].notna()],
    "synthetic_only": measured_routed[
        measured_routed["task_family"].str.startswith("synthetic") | (measured_routed["task_family"] == "chat_shared_prefix")
    ],
}.items():
    summary_rows.append(
        {
            "scope": scope_name,
            "workloads": int(scope_df["workload_name"].nunique()),
            "mean_latency_speedup_vs_fp16": float(scope_df["latency_speedup_vs_fp16"].mean()),
            "mean_energy_ratio_vs_fp16": float(scope_df["energy_ratio_vs_fp16"].mean()),
            "mean_throughput_ratio_vs_fp16": float(scope_df["throughput_ratio_vs_fp16"].mean()),
            "mean_selected_accuracy_pct": (
                float(scope_df["selected_accuracy_pct"].mean())
                if scope_df["selected_accuracy_pct"].notna().any()
                else np.nan
            ),
            "mean_baseline_accuracy_pct": (
                float(scope_df["baseline_accuracy_pct"].mean())
                if scope_df["baseline_accuracy_pct"].notna().any()
                else np.nan
            ),
        }
    )

controller_summary_df = pd.DataFrame(summary_rows)
mode_counts_df = routed["selected_mode"].value_counts().rename_axis("selected_mode").reset_index(name="num_workloads")

print("Controller selection counts:")
display(mode_counts_df)
print("\\nController summary:")
display(controller_summary_df)
""".strip()

EXAMPLES_CODE = """
example_requests_df = pd.DataFrame(
    [
        {
            "request_example": "Hard science MCQ with long prompt and 1-token answer",
            "prompt_tokens": 700,
            "expected_output_tokens": 8,
            "shared_prefix": False,
            "memory_pressure": False,
            "batch_pressure": "normal",
            "task_family": "science_mcq",
        },
        {
            "request_example": "Broad reasoning MCQ with short answer",
            "prompt_tokens": 500,
            "expected_output_tokens": 8,
            "shared_prefix": False,
            "memory_pressure": False,
            "batch_pressure": "normal",
            "task_family": "reasoning_mcq",
        },
        {
            "request_example": "Math generation request",
            "prompt_tokens": 500,
            "expected_output_tokens": 350,
            "shared_prefix": False,
            "memory_pressure": False,
            "batch_pressure": "normal",
            "task_family": "math_generation",
        },
        {
            "request_example": "Shared-prefix chat continuation",
            "prompt_tokens": 1000,
            "expected_output_tokens": 64,
            "shared_prefix": True,
            "memory_pressure": False,
            "batch_pressure": "normal",
            "task_family": "chat_shared_prefix",
        },
        {
            "request_example": "Hot queue / high batch traffic service request",
            "prompt_tokens": 128,
            "expected_output_tokens": 64,
            "shared_prefix": False,
            "memory_pressure": False,
            "batch_pressure": "high",
            "task_family": "serving_batched",
        },
    ]
)
example_requests_df[[
    "controller_phase_label",
    "estimated_prefill_share_pct",
    "selected_mode",
    "route_reason",
]] = example_requests_df.apply(
    lambda row: pd.Series(
        (
            lambda features, classification, decision: (
                classification.label,
                classification.estimated_prefill_share_pct,
                decision.selected_mode_name,
                decision.reason,
            )
        )(
            RequestFeatures(
                prompt_tokens=int(row["prompt_tokens"]),
                expected_output_tokens=int(row["expected_output_tokens"]),
                shared_prefix=bool(row["shared_prefix"]),
                batch_pressure=str(row["batch_pressure"]),
                memory_pressure=bool(row["memory_pressure"]),
                workload_tag=str(row["task_family"]),
            ),
            classify_request(
                RequestFeatures(
                    prompt_tokens=int(row["prompt_tokens"]),
                    expected_output_tokens=int(row["expected_output_tokens"]),
                    shared_prefix=bool(row["shared_prefix"]),
                    batch_pressure=str(row["batch_pressure"]),
                    memory_pressure=bool(row["memory_pressure"]),
                    workload_tag=str(row["task_family"]),
                )
            ),
            route_request(
                RequestFeatures(
                    prompt_tokens=int(row["prompt_tokens"]),
                    expected_output_tokens=int(row["expected_output_tokens"]),
                    shared_prefix=bool(row["shared_prefix"]),
                    batch_pressure=str(row["batch_pressure"]),
                    memory_pressure=bool(row["memory_pressure"]),
                    workload_tag=str(row["task_family"]),
                )
            ),
        )
    ),
    axis=1,
)
example_requests_df
""".strip()

PLOT_CODE = """
if not PLOTTING_AVAILABLE:
    print("Matplotlib not available; skipping plots.")
else:
    plot_df = routed[
        ["workload_name", "selected_mode", "latency_speedup_vs_fp16", "energy_ratio_vs_fp16"]
    ].copy()
    plot_df = plot_df.sort_values("latency_speedup_vs_fp16", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].bar(plot_df["workload_name"], plot_df["latency_speedup_vs_fp16"], color="#4c78a8")
    axes[0].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_title("Recommended controller latency speedup vs FP16")
    axes[0].set_ylabel("Speedup (x)")
    axes[0].tick_params(axis="x", rotation=70)

    axes[1].bar(plot_df["workload_name"], plot_df["energy_ratio_vs_fp16"], color="#f58518")
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1].set_title("Recommended controller energy ratio vs FP16")
    axes[1].set_ylabel("Selected / FP16")
    axes[1].tick_params(axis="x", rotation=70)

    plt.tight_layout()
    plt.show()
""".strip()

PSEUDO_MARKDOWN = """
## Recommended fixed-mode controller

```python
if batch_pressure == "high":
    mode = "int8_plus_continuous_batching"
elif shared_prefix:
    mode = "gptq_plus_prefix_caching"
elif predicted_prefill_share_pct > 40:
    mode = "fp16_baseline"
elif memory_pressure:
    mode = "prefix_caching"
elif predicted_phase == "decode_heavy":
    mode = "gptq_4bit"
else:
    mode = "prefix_caching"
```

This keeps the whole request on **one fixed mode** and uses only signals available before generation starts. It is still phase-aware because the decision is driven by the predicted dominant phase, but it never swaps modes mid-request.
""".strip()

LIMITATIONS_MARKDOWN = """
## Coverage gaps and follow-up reruns

- The user-curated tables are treated as the authoritative results because some notebook outputs were clearly small-sample or earlier intermediate views.
- The notebook compares controller-picked modes against the reported accuracy tables, but it does not itself rerun the expensive full benchmark sweep. The actual harness-side rerun should now happen through `controller_v1`.
- `controller_v1` is wired into the Python benchmarking code, but a fresh end-to-end sweep still needs to be launched to populate new raw JSON / CSV artifacts for the controller rows.
- Judge-quality metrics for `mt_bench_eval` and `alpacaeval2_lc_eval` were not provided in the final written-out table, so the notebook keeps those accuracy discussions qualitative unless a fresh judged rerun is produced.
- The current controller tree intentionally favors the conversation-aligned safe rule for prefill-heavy workloads: route them to FP16 rather than chasing small wins with a riskier branch.
""".strip()

HARNESS_MARKDOWN = """
## Harness integration and accuracy testing

The notebook is the design + analysis layer. The actual benchmark harness now supports a real `controller_v1` mode in Python, so the full accuracy tests should be rerun through the existing pipeline rather than simulated by hand.

Suggested benchmark entrypoint:

```python
from benchmark_modes import run_full_benchmark

results = run_full_benchmark(
    include_hybrids=True,
    repeated_prefix_variants=2,
    test_profile="controller",
)
```

That rerun is what will produce true controller-vs-FP16 accuracy, latency, energy, and memory rows across:

- MMLU-Pro
- GSM8K
- TruthfulQA
- GPQA
- MLU / MMMLU
- MT-Bench
- AlpacaEval 2 LC
- the synthetic workload families
""".strip()


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# ModeSwitch-LLM Fixed-Mode Controller (Updated)\n\n"
            "This notebook replaces the older phase-aware / mid-request switching framing. "
            "The updated direction is:\n\n"
            "- **do not switch modes between prefill and decode**\n"
            "- infer request regime from signals available upfront\n"
            "- send the **entire request** to one fixed mode\n"
            "- optimize latency, energy, and throughput while keeping quality close to the FP16 baseline"
        ),
        nbf.v4.new_markdown_cell(
            "## What changed from the conversation\n\n"
            "- The old direction was separate prefill-mode and decode-mode routing.\n"
            "- The new direction is still phase-aware, but it commits to **one fixed mode per request**.\n"
            "- The controller uses only cheap static request signals before inference starts.\n"
            "- Accuracy is still judged against the FP16 baseline using the same benchmark families Aman kept in the pipeline."
        ),
        nbf.v4.new_markdown_cell(
            "## Provenance and benchmark suite\n\n"
            "Primary source for the numbers in this notebook: the manually curated benchmark tables "
            "supplied on **2026-04-30**.\n\n"
            "Secondary sources used to fill gaps or validate winners:\n\n"
            "- `final_test (7) (2).ipynb` notebook outputs\n"
            "- the screenshots / plots shared in the chat\n\n"
            "When values conflict, the **manually curated table wins**. The notebook extraction is "
            "only used to recover missing benchmark-safe INT8 rows and winner summaries.\n\n"
            "Benchmarks retained from the screenshots:\n\n"
            "1. MMLU-Pro for broad knowledge + reasoning\n"
            "2. GSM8K for math reasoning\n"
            "3. TruthfulQA for factual reliability\n"
            "4. GPQA for harder science / reasoning\n"
            "5. MT-Bench + AlpacaEval 2 LC for chat quality\n"
            "6. MLU / MMMLU for multilingual understanding\n\n"
            "Model-card reference points noted from the screenshot for Llama 3.1 8B Instruct: "
            "roughly **MMLU-Pro 48.3**, **GPQA 30.4**, **GSM8K 84.5**. These are not used as "
            "controller targets directly, but they justify the benchmark mix."
        ),
        nbf.v4.new_code_cell(IMPORTS_CODE),
        nbf.v4.new_code_cell(DATA_CODE),
        nbf.v4.new_code_cell(HELPERS_CODE),
        nbf.v4.new_code_cell(EXTRACT_CODE),
        nbf.v4.new_markdown_cell(
            "## Workload signals\n\n"
            "The controller is allowed to use only signals that are available before generation "
            "starts: prompt length, expected output length, whether a shared prefix exists, memory "
            "pressure, batch pressure, and optionally a lightweight workload tag if the product already has one.\n\n"
            "Phase labels below are derived from the **FP16 baseline prefill share**, because the "
            "project's central story is that the incoming workload regime drives the correct fixed mode. "
            "Those measured phase labels are shown beside the controller's own predicted phase."
        ),
        nbf.v4.new_code_cell('workload_df.sort_values(["workload_name"]).reset_index(drop=True)'),
        nbf.v4.new_markdown_cell(
            "## Mode profiles\n\n"
            "These are the modes we keep in the updated controller story. The critical distinction "
            "is between **benchmark-safe** modes and **aggressive throughput / latency** modes."
        ),
        nbf.v4.new_code_cell("mode_profile_df"),
        nbf.v4.new_markdown_cell(PSEUDO_MARKDOWN),
        nbf.v4.new_code_cell(ROUTE_CODE),
        nbf.v4.new_markdown_cell(HARNESS_MARKDOWN),
        nbf.v4.new_markdown_cell(
            "## Evidence-backed routing table\n\n"
            "This is the core output of the updated controller notebook: one fixed mode per workload "
            "family, with no separate prefill and decode routing. The table also shows whether the "
            "controller's predicted phase matches the measured FP16 phase split."
        ),
        nbf.v4.new_code_cell(SUMMARY_CODE),
        nbf.v4.new_markdown_cell(
            "## Example request-level routes\n\n"
            "The measured workloads do not include a real high-queue serving trace, so the serving "
            "branch is shown here with hypothetical request examples. That keeps the controller "
            "faithful to the new product direction while still grounding the other branches in measured data."
        ),
        nbf.v4.new_code_cell(EXAMPLES_CODE),
        nbf.v4.new_code_cell(PLOT_CODE),
        nbf.v4.new_markdown_cell(LIMITATIONS_MARKDOWN),
    ]
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    }
    return nb


def main() -> None:
    notebook = build_notebook()
    output_path = Path("fixed_mode_controller_updated.ipynb")
    nbf.write(notebook, output_path)
    print(f"Wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
