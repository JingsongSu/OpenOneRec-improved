import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
import json
from vllm import LLM
from vllm.sampling_params import BeamSearchParams

from multiprocessing import Process, Queue, set_start_method

parser = argparse.ArgumentParser()
parser.add_argument('--step', type=int, default=8710, help='global step number (overrides auto-extraction)')
parser.add_argument('--gpu_num', type=int, default=8, help='gpu nums)')
parser.add_argument('--mode', type=str, default='beam', choices=['sampling', 'beam'], help='生成模式：sampling（采样）或 beam（束搜索）')
args = parser.parse_args()

model_step = args.step
MODEL_PATH = f"/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-main/pretrain/model_output/sft/step10000/global_step10000/converted"

DATA_PATH = "/home/jovyan/ceph-1/zhangguozhu/generative_recommendation/OpenOneRec_data/output/eval/sft/sft_video_rec.parquet"

NUM_GPUS = args.gpu_num
BATCH_SIZE = 128
BEAM_WIDTH = 100
MAX_TOKENS = 6


def convert_messages(messages, add_think_pattern=False):
    msg_list = []
    for msg in messages:
        content = msg['content']
        if isinstance(content, str):
            msg_list.append({
                'role': msg['role'],
                'content': content
            })
        elif isinstance(content, dict) and 'type' in content and content['type'] == 'text':
            msg_list.append({
                'role': msg['role'],
                'content': content['text']
            })
        elif isinstance(content, list) and len(content) > 0:
            content_text = ""
            for c in content:
                if isinstance(c, dict) and 'type' in c and c['type'] == 'text':
                    content_text += c['text']
                elif isinstance(c, str):
                    content_text += c
                else:
                    continue
            msg_list.append({
                'role': msg['role'],
                'content': content_text
            })
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    if add_think_pattern:
        # Process thinking pattern: add /think or /no_think suffix to user messages
        # based on whether assistant message contains reasoning content
        for i in range(len(msg_list)):
            if msg_list[i]['role'] == 'assistant':
                assistant_content = msg_list[i]['content']

                # Find corresponding user message (typically the previous one)
                user_idx = i - 1
                if user_idx < 0 or msg_list[user_idx]['role'] != 'user':
                    continue

                # Check if assistant content contains <think> tags
                pattern = r'<think>(.*?)</think>'
                match = re.search(pattern, assistant_content, re.DOTALL)

                if match is None:
                    # No reasoning tags found: add empty tags and mark as /no_think
                    msg_list[user_idx]['content'] += "/no_think"
                    msg_list[i]['content'] = "<think>\n</think>\n" + assistant_content
                else:
                    # Reasoning tags found: check if they contain actual content
                    reasoning_content = match.group(1)
                    if reasoning_content.strip():
                        # Has reasoning content: mark as /think
                        msg_list[user_idx]['content'] += "/think"
                    else:
                        # Empty reasoning tags: mark as /no_think
                        msg_list[user_idx]['content'] += "/no_think"

    return msg_list




def run_batch(llm, params, prompt_list, answer_list, len_list):
    hit = 0
    cnt = 0
    
    outputs = llm.beam_search(prompt_list, params)

    for i, output in enumerate(outputs):
        cnt += 1
        for seq in output.sequences:
            generated_text = seq.text
            res = generated_text[len_list[i]:]
            if res == answer_list[i]:
                hit += 1
                break

    return hit, cnt


def evaluate_worker(rank, world_size, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    print(f"[Rank {rank}] Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    print(f"[Rank {rank}] Loading vLLM...")

    llm = LLM(
        model=MODEL_PATH,
        max_logprobs=512,
        gpu_memory_utilization=0.9,
    )

    params = BeamSearchParams(
        beam_width=BEAM_WIDTH,
        max_tokens=MAX_TOKENS,
    )

    df = pd.read_parquet(DATA_PATH)

    # 数据切分
    df = df.iloc[rank::world_size].reset_index(drop=True)
    print(f"[Rank {rank}] Assigned {len(df)} samples")

    hit = 0
    cnt = 0

    prompt_list = []
    answer_list = []
    len_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"GPU-{rank}",):
        messages_ = json.loads(row["messages"])
        messages = messages_[:-1]
        answer = messages_[-1]["content"][0]["text"]

        msg_converted = convert_messages(messages)

        formatted_prompt = tokenizer.apply_chat_template(
            msg_converted,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        prompt_list.append({"prompt": formatted_prompt})
        answer_list.append(answer)
        len_list.append(len(formatted_prompt))

        if len(prompt_list) >= BATCH_SIZE:
            batch_hit, batch_cnt = run_batch(llm, params, prompt_list, answer_list, len_list)

            hit += batch_hit
            cnt += batch_cnt

            prompt_list = []
            answer_list = []
            len_list = []

    # flush最后一批
    if len(prompt_list) > 0:
        batch_hit, batch_cnt = run_batch(llm, params, prompt_list, answer_list, len_list)

        hit += batch_hit
        cnt += batch_cnt

    result_queue.put({
        "rank": rank,
        "hit": hit,
        "cnt": cnt,
    })

    print(f"[Rank {rank}] Hit={hit}, Cnt={cnt}, Ratio={hit/cnt:.6f}")


def main():
    set_start_method("spawn", force=True)
    
    result_queue = Queue()

    processes = []

    for rank in range(NUM_GPUS):
        p = Process(target=evaluate_worker, args=(rank, NUM_GPUS, result_queue,))
        p.start()
        processes.append(p)

    total_hit = 0
    total_cnt = 0

    for _ in range(NUM_GPUS):
        result = result_queue.get()
        print(f"Receive rank {result['rank']} result: {result['hit']}/{result['cnt']}")

        total_hit += result["hit"]
        total_cnt += result["cnt"]

    for p in processes:
        p.join()

    print("=" * 50)
    print("FINAL RESULT")
    print(f"Hit Ratio = {total_hit}/{total_cnt} = {total_hit/total_cnt:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
