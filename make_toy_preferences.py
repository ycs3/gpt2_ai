import argparse
import json
from pathlib import Path


DEFAULT_CHOSEN_TEMPLATES = [
    "Here is a clear, concise answer: {prompt}",
    "A helpful response would be: {prompt}",
    "One good answer is: {prompt}",
]

DEFAULT_REJECTED_TEMPLATES = [
    "I do not know. {prompt}",
    "Maybe something about this: {prompt}",
    "This is a weak answer for: {prompt}",
]


def load_prompts(path):
    prompts = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                prompt = row.get("prompt", "").strip()
                if prompt:
                    prompts.append(prompt)
            else:
                prompts.append(line)
    return prompts


def build_pair(prompt, index):
    chosen_template = DEFAULT_CHOSEN_TEMPLATES[index % len(DEFAULT_CHOSEN_TEMPLATES)]
    rejected_template = DEFAULT_REJECTED_TEMPLATES[index % len(DEFAULT_REJECTED_TEMPLATES)]
    return {
        "prompt": prompt,
        "chosen": chosen_template.format(prompt=prompt.lower()),
        "rejected": rejected_template.format(prompt=prompt.lower()),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate a toy preferences.jsonl starter set from a prompt list.")
    parser.add_argument("--prompts", type=str, default="prompts.txt", help="Input prompt list or JSONL with prompt fields.")
    parser.add_argument("--output", type=str, default="preferences.generated.jsonl", help="Output JSONL file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file instead of appending.")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    if not prompts:
        raise ValueError(f"no prompts found in {args.prompts}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"

    written = 0
    with open(output_path, mode, encoding="utf-8") as handle:
        for idx, prompt in enumerate(prompts):
            row = build_pair(prompt, idx)
            handle.write(json.dumps(row) + "\n")
            written += 1

    print(f"wrote {written} toy preference pairs to {output_path}")


if __name__ == "__main__":
    main()
