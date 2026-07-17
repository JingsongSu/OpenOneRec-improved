import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
import traceback
from multiprocessing import Process, Queue, set_start_method

import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM
from vllm.sampling_params import BeamSearchParams


parser = argparse.ArgumentParser()

parser.add_argument(
    "--step",
    type=int,
    default=20000,
    help="global step number",
)
parser.add_argument(
    "--gpu_num",
    type=int,
    default=8,
    help="gpu nums",
)
parser.add_argument(
    "--mode",
    type=str,
    default="beam",
    choices=["beam"],
)
parser.add_argument(
    "--print_samples",
    type=int,
    default=5,
    help="Rank 0 打印前多少条样本的推理结果",
)
parser.add_argument(
    "--print_beams",
    type=int,
    default=5,
    help="每条样本打印前多少个 beam",
)
parser.add_argument(
    "--debug_mismatches",
    type=int,
    default=3,
    help="每个 rank 额外打印多少条 mismatch",
)
parser.add_argument(
    "--generate_empty_think",
    action="store_true",
    help=(
        "不把固定空 think 块放入 prefill，而是让模型自行生成。"
        "一般不要启用；启用后会自动把 max_tokens 增加 4。"
    ),
)

args = parser.parse_args()

model_step = args.step

MODEL_PATH = (
    "/home/jovyan/ceph-1/sujinsong/sujinsong/"
    "OpenOneRec-latent/pretrain/model_output/sft_pause_k5-0.2/"
    f"step{model_step}/global_step{model_step}/converted"
)

DATA_PATH = (
    "/home/jovyan/ceph-1/zhangguozhu/"
    "generative_recommendation/OpenOneRec_data/output/eval/sft/"
    "sft_video_rec.parquet"
)

NUM_GPUS = args.gpu_num

BATCH_SIZE = 128
BEAM_WIDTH = 100

# 标准答案固定为：
# <|sid_begin|> + s_a + s_b + s_c + s_d + <|sid_end|>
SID_MAX_TOKENS = 6

# Qwen3 在 non-thinking 模式下使用的固定空思考块。
EMPTY_THINK_TEXT = "<think>\n\n</think>\n\n"


def convert_messages(messages):
    msg_list = []

    for msg in messages:
        content = msg["content"]

        if isinstance(content, str):
            msg_list.append(
                {
                    "role": msg["role"],
                    "content": content,
                }
            )

        elif (
            isinstance(content, dict)
            and content.get("type") == "text"
        ):
            msg_list.append(
                {
                    "role": msg["role"],
                    "content": content.get("text", ""),
                }
            )

        elif isinstance(content, list):
            content_text = ""

            for item in content:
                if isinstance(item, str):
                    content_text += item
                elif (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                ):
                    content_text += item.get("text", "")
                elif (
                    isinstance(item, dict)
                    and "text" in item
                ):
                    content_text += str(item["text"])

            msg_list.append(
                {
                    "role": msg["role"],
                    "content": content_text,
                }
            )

        else:
            raise ValueError(
                f"Unsupported content type: {type(content)}"
            )

    return msg_list


def extract_answer_text(message):
    content = message["content"]

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        if "text" in content:
            return str(content["text"])

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif (
                isinstance(item, dict)
                and item.get("type") == "text"
            ):
                parts.append(item.get("text", ""))
            elif (
                isinstance(item, dict)
                and "text" in item
            ):
                parts.append(str(item["text"]))

        return "".join(parts)

    raise ValueError(
        f"Unsupported answer content type: {type(content)}"
    )


def load_pause_config(tokenizer):
    config = AutoConfig.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    pause_token = getattr(
        config,
        "pause_token",
        "<|latent_pause|>",
    )
    pause_count = int(
        getattr(
            config,
            "pause_count",
            5,
        )
    )

    pause_token_ids = tokenizer.encode(
        pause_token,
        add_special_tokens=False,
    )

    if len(pause_token_ids) != 1:
        raise ValueError(
            f"Pause token {pause_token!r} should map to one token, "
            f"but got {pause_token_ids}. "
            "Please run tools/finalize_pause_checkpoint.py first."
        )

    pause_token_id = int(pause_token_ids[0])

    config_pause_token_id = getattr(
        config,
        "pause_token_id",
        None,
    )

    if (
        config_pause_token_id is not None
        and int(config_pause_token_id) != pause_token_id
    ):
        raise ValueError(
            "Pause token ID mismatch: "
            f"config={config_pause_token_id}, "
            f"tokenizer={pause_token_id}"
        )

    if pause_count <= 0:
        raise ValueError(
            f"pause_count must be positive, got {pause_count}"
        )

    repeated_ids = tokenizer.encode(
        pause_token * pause_count,
        add_special_tokens=False,
    )
    expected_ids = [pause_token_id] * pause_count

    if repeated_ids != expected_ids:
        raise ValueError(
            "Repeated pause token encoding mismatch: "
            f"expected={expected_ids}, got={repeated_ids}"
        )

    if int(config.vocab_size) != len(tokenizer):
        raise ValueError(
            "Tokenizer/config vocabulary mismatch: "
            f"config.vocab_size={config.vocab_size}, "
            f"len(tokenizer)={len(tokenizer)}"
        )

    empty_think_ids = tokenizer.encode(
        EMPTY_THINK_TEXT,
        add_special_tokens=False,
    )

    if not empty_think_ids:
        raise ValueError(
            "EMPTY_THINK_TEXT encoded to an empty token sequence"
        )

    return (
        pause_token,
        pause_token_id,
        pause_count,
        [int(token_id) for token_id in empty_think_ids],
    )


def endswith_ids(token_ids, suffix_ids):
    if not suffix_ids:
        return True

    if len(token_ids) < len(suffix_ids):
        return False

    return token_ids[-len(suffix_ids):] == suffix_ids


def startswith_ids(token_ids, prefix_ids):
    if not prefix_ids:
        return True

    if len(token_ids) < len(prefix_ids):
        return False

    return token_ids[:len(prefix_ids)] == prefix_ids


def build_prompt_token_ids(
    tokenizer,
    messages,
    pause_token_id,
    pause_count,
    empty_think_ids,
    generate_empty_think,
):
    """构建与 Pause SFT 训练顺序一致的推理前缀。

    训练中的实际顺序由现有输出可确认是：
        assistant header
        -> pause * K
        -> <think>\\n\\n</think>\\n\\n
        -> SID

    不同 tokenizer 版本对 add_generation_prompt=True 的行为可能不同：
    有的会在末尾自动放空 think 块，有的只放 assistant header。

    这里统一规范为：
        assistant header + pause * K + empty think

    如果 --generate_empty_think，则只 prefill 到 pause，让模型自行生成空 think。
    """

    msg_converted = convert_messages(messages)

    base_prompt_ids = tokenizer.apply_chat_template(
        msg_converted,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    if hasattr(base_prompt_ids, "tolist"):
        base_prompt_ids = base_prompt_ids.tolist()

    if (
        base_prompt_ids
        and isinstance(base_prompt_ids[0], list)
    ):
        if len(base_prompt_ids) != 1:
            raise ValueError(
                "Unexpected batched result from apply_chat_template"
            )
        base_prompt_ids = base_prompt_ids[0]

    base_prompt_ids = [
        int(token_id)
        for token_id in base_prompt_ids
    ]

    # 某些 Qwen3 chat template 已经在 generation prompt 后添加空 think。
    # 为了把 pause 放到空 think 之前，先移除已有的空 think，再统一重建。
    if endswith_ids(base_prompt_ids, empty_think_ids):
        base_prompt_ids = base_prompt_ids[
            :-len(empty_think_ids)
        ]

    prompt_token_ids = (
        base_prompt_ids
        + [pause_token_id] * pause_count
    )

    if not generate_empty_think:
        prompt_token_ids += empty_think_ids

    return prompt_token_ids


def get_terminal_token_ids(tokenizer):
    terminal_ids = set()

    for token_id in (
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    ):
        if token_id is not None and int(token_id) >= 0:
            terminal_ids.add(int(token_id))

    im_end_id = tokenizer.convert_tokens_to_ids(
        "<|im_end|>"
    )
    unk_id = tokenizer.unk_token_id

    if (
        im_end_id is not None
        and int(im_end_id) >= 0
        and (
            unk_id is None
            or int(im_end_id) != int(unk_id)
        )
    ):
        terminal_ids.add(int(im_end_id))

    return terminal_ids


def strip_trailing_terminal_ids(
    token_ids,
    terminal_ids,
):
    result = [
        int(token_id)
        for token_id in token_ids
    ]

    while (
        result
        and result[-1] in terminal_ids
    ):
        result.pop()

    return result


def strip_leading_empty_think(
    token_ids,
    empty_think_ids,
):
    """容错处理：若模型仍生成了空 think，则从候选前缀移除。"""

    result = [
        int(token_id)
        for token_id in token_ids
    ]

    while startswith_ids(
        result,
        empty_think_ids,
    ):
        result = result[
            len(empty_think_ids):
        ]

    return result


def extract_generated_token_ids(
    sequence,
    prompt_token_ids,
):
    sequence_token_ids = [
        int(token_id)
        for token_id in sequence.tokens
    ]

    prompt_length = len(prompt_token_ids)

    # 常见 vLLM 版本：sequence.tokens 包含 prompt + generated。
    if (
        len(sequence_token_ids) >= prompt_length
        and sequence_token_ids[:prompt_length]
        == prompt_token_ids
    ):
        return sequence_token_ids[
            prompt_length:
        ]

    # 兼容只返回 generated tokens 的版本。
    return sequence_token_ids


def normalize_candidate_ids(
    candidate_ids,
    empty_think_ids,
    terminal_ids,
):
    candidate_ids = strip_leading_empty_think(
        candidate_ids,
        empty_think_ids,
    )

    candidate_ids = strip_trailing_terminal_ids(
        candidate_ids,
        terminal_ids,
    )

    return candidate_ids


def run_batch(
    llm,
    params,
    prompt_list,
    prompt_token_id_list,
    answer_token_id_list,
    answer_text_list,
    tokenizer,
    empty_think_ids,
    rank,
    print_budget,
    print_beams,
    mismatch_budget,
    processed_samples,
):
    hit = 0
    cnt = 0

    outputs = llm.beam_search(
        prompt_list,
        params,
    )

    terminal_ids = get_terminal_token_ids(
        tokenizer
    )

    for i, output in enumerate(outputs):
        cnt += 1

        expected_ids = strip_trailing_terminal_ids(
            answer_token_id_list[i],
            terminal_ids,
        )

        candidates = []

        for sequence in output.sequences:
            raw_generated_ids = (
                extract_generated_token_ids(
                    sequence,
                    prompt_token_id_list[i],
                )
            )

            normalized_ids = normalize_candidate_ids(
                raw_generated_ids,
                empty_think_ids,
                terminal_ids,
            )

            candidates.append(
                {
                    "raw_ids": raw_generated_ids,
                    "ids": normalized_ids,
                }
            )

        matched = any(
            candidate["ids"] == expected_ids
            for candidate in candidates
        )

        if matched:
            hit += 1

        should_print_sample = (
            rank == 0
            and print_budget > 0
        )

        if should_print_sample:
            print_budget -= 1

            print("\n" + "=" * 100)
            print(
                f"[Rank {rank}] inference sample "
                f"{processed_samples + i}"
            )
            print(f"matched={matched}")
            print(
                f"prompt_length="
                f"{len(prompt_token_id_list[i])}"
            )
            print(
                f"prompt_tail="
                f"{prompt_token_id_list[i][-16:]}"
            )
            print(
                f"empty_think_ids="
                f"{empty_think_ids}"
            )
            print(
                f"answer_text="
                f"{answer_text_list[i]!r}"
            )
            print(
                f"answer_ids="
                f"{expected_ids}"
            )

            beam_count = min(
                print_beams,
                len(candidates),
            )

            for beam_index in range(beam_count):
                candidate = candidates[beam_index]

                raw_text = tokenizer.decode(
                    candidate["raw_ids"],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )

                normalized_text = tokenizer.decode(
                    candidate["ids"],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )

                print(
                    f"beam[{beam_index}] "
                    f"match="
                    f"{candidate['ids'] == expected_ids}"
                )
                print(
                    f"beam[{beam_index}] "
                    f"raw_ids={candidate['raw_ids']}"
                )
                print(
                    f"beam[{beam_index}] "
                    f"raw_text={raw_text!r}"
                )
                print(
                    f"beam[{beam_index}] "
                    f"normalized_ids={candidate['ids']}"
                )
                print(
                    f"beam[{beam_index}] "
                    f"normalized_text="
                    f"{normalized_text!r}"
                )

            print("=" * 100 + "\n")

        elif (
            not matched
            and mismatch_budget > 0
        ):
            mismatch_budget -= 1

            top_candidate = (
                candidates[0]
                if candidates
                else {
                    "raw_ids": [],
                    "ids": [],
                }
            )

            candidate_text = tokenizer.decode(
                top_candidate["ids"],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            print(
                f"[Rank {rank} Mismatch]\n"
                f"  expected_ids={expected_ids}\n"
                f"  candidate_raw_ids="
                f"{top_candidate['raw_ids']}\n"
                f"  candidate_ids="
                f"{top_candidate['ids']}\n"
                f"  expected_text="
                f"{answer_text_list[i]!r}\n"
                f"  candidate_text="
                f"{candidate_text!r}\n"
            )

    return (
        hit,
        cnt,
        print_budget,
        mismatch_budget,
    )


def evaluate_worker(
    rank,
    world_size,
    result_queue,
):
    try:
        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = str(rank)

        print(
            f"[Rank {rank}] Loading tokenizer..."
        )

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
        )

        (
            pause_token,
            pause_token_id,
            pause_count,
            empty_think_ids,
        ) = load_pause_config(tokenizer)

        print(
            f"[Rank {rank}] Pause config: "
            f"token={pause_token!r}, "
            f"token_id={pause_token_id}, "
            f"count={pause_count}"
        )
        print(
            f"[Rank {rank}] Empty think: "
            f"text={EMPTY_THINK_TEXT!r}, "
            f"ids={empty_think_ids}"
        )
        print(
            f"[Rank {rank}] "
            f"generate_empty_think="
            f"{args.generate_empty_think}"
        )

        max_tokens = SID_MAX_TOKENS

        if args.generate_empty_think:
            max_tokens += len(
                empty_think_ids
            )

        print(
            f"[Rank {rank}] "
            f"beam max_tokens={max_tokens}"
        )

        print(
            f"[Rank {rank}] Loading vLLM..."
        )

        llm = LLM(
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            trust_remote_code=True,
            max_logprobs=512,
            gpu_memory_utilization=0.9,
        )

        params = BeamSearchParams(
            beam_width=BEAM_WIDTH,
            max_tokens=max_tokens,
        )

        df = pd.read_parquet(
            DATA_PATH
        )

        df = df.iloc[
            rank::world_size
        ].reset_index(
            drop=True
        )

        print(
            f"[Rank {rank}] Assigned "
            f"{len(df)} samples"
        )

        hit = 0
        cnt = 0
        processed_samples = 0

        print_budget = (
            args.print_samples
            if rank == 0
            else 0
        )
        mismatch_budget = (
            args.debug_mismatches
        )

        prompt_list = []
        prompt_token_id_list = []
        answer_token_id_list = []
        answer_text_list = []

        printed_prompt_check = False

        def flush_batch():
            nonlocal hit
            nonlocal cnt
            nonlocal processed_samples
            nonlocal print_budget
            nonlocal mismatch_budget
            nonlocal prompt_list
            nonlocal prompt_token_id_list
            nonlocal answer_token_id_list
            nonlocal answer_text_list

            if not prompt_list:
                return

            (
                batch_hit,
                batch_cnt,
                print_budget,
                mismatch_budget,
            ) = run_batch(
                llm=llm,
                params=params,
                prompt_list=prompt_list,
                prompt_token_id_list=(
                    prompt_token_id_list
                ),
                answer_token_id_list=(
                    answer_token_id_list
                ),
                answer_text_list=(
                    answer_text_list
                ),
                tokenizer=tokenizer,
                empty_think_ids=(
                    empty_think_ids
                ),
                rank=rank,
                print_budget=print_budget,
                print_beams=args.print_beams,
                mismatch_budget=(
                    mismatch_budget
                ),
                processed_samples=(
                    processed_samples
                ),
            )

            hit += batch_hit
            cnt += batch_cnt
            processed_samples += batch_cnt

            prompt_list = []
            prompt_token_id_list = []
            answer_token_id_list = []
            answer_text_list = []

        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"GPU-{rank}",
        ):
            raw_messages = row["messages"]

            if isinstance(
                raw_messages,
                str,
            ):
                messages_all = json.loads(
                    raw_messages
                )
            elif isinstance(
                raw_messages,
                list,
            ):
                messages_all = raw_messages
            else:
                raise ValueError(
                    "Unsupported messages "
                    "column type: "
                    f"{type(raw_messages)}"
                )

            if len(messages_all) < 2:
                raise ValueError(
                    "Evaluation sample must "
                    "contain a target message"
                )

            messages = messages_all[:-1]

            answer = extract_answer_text(
                messages_all[-1]
            )

            prompt_token_ids = (
                build_prompt_token_ids(
                    tokenizer=tokenizer,
                    messages=messages,
                    pause_token_id=(
                        pause_token_id
                    ),
                    pause_count=(
                        pause_count
                    ),
                    empty_think_ids=(
                        empty_think_ids
                    ),
                    generate_empty_think=(
                        args.generate_empty_think
                    ),
                )
            )

            answer_token_ids = tokenizer.encode(
                answer,
                add_special_tokens=False,
            )
            answer_token_ids = [
                int(token_id)
                for token_id in answer_token_ids
            ]

            if not answer_token_ids:
                raise ValueError(
                    "Empty answer token IDs: "
                    f"{answer!r}"
                )

            if (
                len(answer_token_ids)
                > SID_MAX_TOKENS
            ):
                raise ValueError(
                    "Ground-truth answer is "
                    "longer than SID_MAX_TOKENS: "
                    f"answer_len="
                    f"{len(answer_token_ids)}, "
                    f"SID_MAX_TOKENS="
                    f"{SID_MAX_TOKENS}, "
                    f"answer={answer!r}"
                )

            if not printed_prompt_check:
                printed_prompt_check = True

                if args.generate_empty_think:
                    expected_tail = (
                        [pause_token_id]
                        * pause_count
                    )
                else:
                    expected_tail = (
                        [pause_token_id]
                        * pause_count
                        + empty_think_ids
                    )

                actual_tail = (
                    prompt_token_ids[
                        -len(expected_tail):
                    ]
                )

                print(
                    f"[Rank {rank}] "
                    f"Prompt check:\n"
                    f"  prompt_length="
                    f"{len(prompt_token_ids)}\n"
                    f"  actual_tail="
                    f"{actual_tail}\n"
                    f"  expected_tail="
                    f"{expected_tail}\n"
                    f"  answer_ids="
                    f"{answer_token_ids}\n"
                    f"  answer_text="
                    f"{answer!r}"
                )

                if actual_tail != expected_tail:
                    raise ValueError(
                        "Prompt tail does not "
                        "match expected "
                        "pause/think prefix"
                    )

            prompt_list.append(
                {
                    "prompt_token_ids":
                    prompt_token_ids,
                }
            )
            prompt_token_id_list.append(
                prompt_token_ids
            )
            answer_token_id_list.append(
                answer_token_ids
            )
            answer_text_list.append(
                answer
            )

            if (
                len(prompt_list)
                >= BATCH_SIZE
            ):
                flush_batch()

        flush_batch()

        result_queue.put(
            {
                "ok": True,
                "rank": rank,
                "hit": hit,
                "cnt": cnt,
            }
        )

        ratio = (
            hit / cnt
            if cnt > 0
            else 0.0
        )

        print(
            f"[Rank {rank}] "
            f"Hit={hit}, "
            f"Cnt={cnt}, "
            f"Ratio={ratio:.6f}"
        )

    except Exception:
        error = traceback.format_exc()

        result_queue.put(
            {
                "ok": False,
                "rank": rank,
                "error": error,
            }
        )

        print(
            f"[Rank {rank}] "
            f"Failed:\n{error}"
        )


def main():
    if not os.path.isdir(
        MODEL_PATH
    ):
        raise FileNotFoundError(
            "MODEL_PATH does not exist: "
            f"{MODEL_PATH}"
        )

    if not os.path.isfile(
        DATA_PATH
    ):
        raise FileNotFoundError(
            "DATA_PATH does not exist: "
            f"{DATA_PATH}"
        )

    set_start_method(
        "spawn",
        force=True,
    )

    result_queue = Queue()
    processes = []

    for rank in range(
        NUM_GPUS
    ):
        process = Process(
            target=evaluate_worker,
            args=(
                rank,
                NUM_GPUS,
                result_queue,
            ),
        )

        process.start()
        processes.append(
            process
        )

    total_hit = 0
    total_cnt = 0
    errors = []

    for _ in range(
        NUM_GPUS
    ):
        result = result_queue.get()

        if not result.get(
            "ok",
            False,
        ):
            errors.append(
                f"Rank "
                f"{result['rank']}:\n"
                f"{result['error']}"
            )
            continue

        print(
            f"Receive rank "
            f"{result['rank']} result: "
            f"{result['hit']}/"
            f"{result['cnt']}"
        )

        total_hit += result["hit"]
        total_cnt += result["cnt"]

    for process in processes:
        process.join()

    bad_exit_codes = [
        (
            rank,
            process.exitcode,
        )
        for rank, process in enumerate(
            processes
        )
        if process.exitcode not in (
            0,
            None,
        )
    ]

    if bad_exit_codes:
        errors.append(
            "Non-zero worker exit codes: "
            f"{bad_exit_codes}"
        )

    if errors:
        raise RuntimeError(
            "Evaluation worker failed:\n\n"
            + "\n\n".join(errors)
        )

    total_ratio = (
        total_hit / total_cnt
        if total_cnt > 0
        else 0.0
    )

    print("=" * 50)
    print("FINAL RESULT")
    print(
        f"Hit Ratio = "
        f"{total_hit}/"
        f"{total_cnt} = "
        f"{total_ratio:.6f}"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
