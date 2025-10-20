import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import vllm
from vllm.lora.request import LoRARequest


def batched(l, n):
    """Batch a list into chunks of size n"""
    num_batches = (len(l) + n - 1) // n  # Ceiling division
    return [l[i * n : (i + 1) * n] for i in range(num_batches)]


def build_test_message(sample):
    """Build the chat message for a test sample"""
    # TODO: Implement based on your specific prompt format
    # Example implementation:
    return [
        {"role": "system", "content": "You are a content moderator."},
        {
            "role": "user",
            "content": f"Rule: {sample.rule}\nComment: {sample.comment}\nDoes this comment violate the rule? Answer yes or no.",
        },
    ]


os.environ["VLLM_USE_V1"] = "0"
BASE_MODEL_PATH = "/kaggle/input/qwen-3/transformers/0.6b/1"


def run_inference_on_device(df, model_str):
    llm = vllm.LLM(
        BASE_MODEL_PATH,
        quantization=None,  # "gptq",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=2048,
        disable_log_stats=True,
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=64,
    )

    tokenizer = llm.get_tokenizer()

    texts = [build_test_message(sample) for sample in df.itertuples()]
    print(texts[:2])
    outputs = llm.generate(
        [tokenizer.apply_chat_template(text, tokenize=False) for text in texts],
        vllm.SamplingParams(
            skip_special_tokens=True,
            max_tokens=10,
            logprobs=2,
        ),
        use_tqdm=True,
        lora_request=LoRARequest("default", 1, model_str),
    )

    answers = []
    for out in outputs:
        sample_lp = {}
        for lp in out.outputs[0].logprobs:
            if 2152 in lp:
                sample_lp["no"] = lp[2152].logprob
            if 9693 in lp:
                sample_lp["yes"] = lp[9693].logprob
        answers.append(sample_lp)

    yes_logprobs = []
    no_logprobs = []

    for lp_dict in answers:
        # Get log probabilities, default to very negative value if token not in top-k
        yes_lp = lp_dict.get("yes", lp_dict.get("Yes", -100.0))
        no_lp = lp_dict.get("no", lp_dict.get("No", -100.0))

        yes_logprobs.append(yes_lp)
        no_logprobs.append(no_lp)

    print(f"yes: {yes_logprobs}")

    yes_probs = np.exp(yes_logprobs)
    no_probs = np.exp(no_logprobs)
    total_probs = yes_probs + no_probs

    # Probability of "yes" (positive class)
    prob_yes = yes_probs / total_probs

    df = df.copy()
    df["rule_violation"] = prob_yes
    return df


def worker(device_id, df_slice, model_str, return_dict):
    """Worker function that runs on a specific GPU device"""
    # Limit this process to only see one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    print(f"[Worker {device_id}] Running on GPU {device_id}, data size={len(df_slice)}")

    preds = run_inference_on_device(df_slice, model_str)
    return_dict[device_id] = preds
    print(f"[Worker {device_id}] Finished, stored results")


def main():
    # Set multiprocessing start method for CUDA compatibility
    mp.set_start_method("spawn", force=True)

    test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
    rules = {
        i: rule
        for i, rule in enumerate(
            sorted(pd.concat([train_df, test_df])["rule"].unique())
        )
    }

    manager = mp.Manager()
    return_dict = manager.dict()

    tasks = [(i, rule) for i, rule in rules.items()]

    for batch in batched(tasks, 2):
        processes = []
        for i, (rule_idx, rule) in enumerate(batch):
            model_str = f"/kaggle/working/model_{rule_idx}.pt"
            rule_df = test_df[test_df["rule"] == rule].copy()

            # Pass all required arguments including model_str
            p = mp.Process(target=worker, args=(i, rule_df, model_str, return_dict))
            processes.append(p)
            p.start()
            print(f"Starting process for rule {rule_idx} on GPU {i}")

        for p in processes:
            p.join()
            print("Finished process")

    # Merge results
    print(f"Total results in return_dict: {len(return_dict)}")
    predictions = pd.concat([df for df in return_dict.values()], ignore_index=True)

    # Merge predictions with test_df
    test_df = test_df.merge(
        predictions[["row_id", "rule_violation"]], on="row_id", how="left"
    )
    test_df[["row_id", "rule_violation"]].to_csv(
        "/kaggle/working/submission.csv", index=False
    )
    print("Submission saved to /kaggle/working/submission.csv")


if __name__ == "__main__":
    main()
