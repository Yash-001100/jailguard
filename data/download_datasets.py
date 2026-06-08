"""
Downloads additional HuggingFace datasets into Dataset/
Run with: python data/download_datasets.py
"""
from datasets import load_dataset
import pandas as pd
import os

SAFE_DIR = os.path.join(os.path.dirname(__file__), "..", "Dataset", "Safe Dataset")
UNSAFE_DIR = os.path.join(os.path.dirname(__file__), "..", "Dataset", "Unsafe Dataset")
os.makedirs(SAFE_DIR, exist_ok=True)
os.makedirs(UNSAFE_DIR, exist_ok=True)

LARGE_DATASET_CAP = 30_000  # max rows to keep from 1M+ datasets


def save(df, path):
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  Saved {len(df)} rows -> {os.path.basename(path)}")


def download_beavertails():
    """~330k conversations with safety labels and category."""
    print("\n[1/7] BeaverTails (PKU-Alignment/BeaverTails)")
    ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
    df = ds.to_pandas()[["prompt", "response", "category", "is_safe"]]

    unsafe = df[df["is_safe"] == False][["prompt", "category"]].rename(columns={"category": "attack_type"})
    safe = df[df["is_safe"] == True][["prompt"]].assign(label=0)

    save(unsafe, os.path.join(UNSAFE_DIR, "beavertails_unsafe.csv"))
    save(safe.sample(min(len(safe), 20_000), random_state=42),
         os.path.join(SAFE_DIR, "beavertails_safe.csv"))


def download_salad():
    """Adversarially enhanced harmful questions with categories."""
    print("\n[2/7] Salad-Data (OpenSafetyLab/Salad-Data)")
    ds = load_dataset("OpenSafetyLab/Salad-Data", "attack_enhanced_set", split="train")
    df = ds.to_pandas()
    print(f"  Columns: {list(df.columns)}")

    # Actual columns: baseq, augq, 2-category, 1-category, 3-category, aid, qid, method
    text_col = "augq" if "augq" in df.columns else "baseq"
    cat_col = "2-category" if "2-category" in df.columns else None

    out = pd.DataFrame({"prompt": df[text_col].astype(str)})
    if cat_col:
        out["attack_type"] = df[cat_col].astype(str)
    save(out, os.path.join(UNSAFE_DIR, "salad_attack.csv"))


def download_categorical_harmful():
    """Harmful QA organised by category — great for attack-type labels."""
    print("\n[3/7] CategoricalHarmfulQA (declare-lab/CategoricalHarmfulQA)")
    ds = load_dataset("declare-lab/CategoricalHarmfulQA", split="en")
    df = ds.to_pandas()
    print(f"  Columns: {list(df.columns)}")

    text_col = next((c for c in ["question", "prompt", "text"] if c in df.columns), df.columns[0])
    cat_col = next((c for c in ["category", "type", "harm_type"] if c in df.columns), None)

    out = pd.DataFrame({"prompt": df[text_col].astype(str)})
    if cat_col:
        out["attack_type"] = df[cat_col].astype(str)
    save(out, os.path.join(UNSAFE_DIR, "categorical_harmful_qa.csv"))


def download_lmsys():
    """1M real user chats — take a safe sample and flag toxic ones."""
    print("\n[4/7] LMSYS-Chat-1M (lmsys/lmsys-chat-1m) — capped at 30k")
    try:
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        rows = []
        for row in ds:
            if len(rows) >= LARGE_DATASET_CAP:
                break
            # Extract first user turn as the prompt
            conv = row.get("conversation", [])
            user_turns = [t["content"] for t in conv if t.get("role") == "user"]
            if user_turns:
                rows.append({
                    "prompt": user_turns[0],
                    "label": 0  # treat as safe (real user chat, mostly benign)
                })
        save(pd.DataFrame(rows), os.path.join(SAFE_DIR, "lmsys_chat_30k.csv"))
    except Exception as e:
        print(f"  FAILED (dataset may be gated — needs HuggingFace login): {e}")
        print("  Run: huggingface-cli login   then retry")


def download_jailbreakbench():
    """~100 canonical jailbreak behaviors with categories."""
    print("\n[5/7] JailbreakBench (JailbreakBench/JBB-Behaviors)")
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    df = ds.to_pandas()
    print(f"  Columns: {list(df.columns)}")

    text_col = next((c for c in ["Goal", "Behavior", "prompt", "text"] if c in df.columns), df.columns[0])
    cat_col = next((c for c in ["Category", "category", "SemanticCategory"] if c in df.columns), None)

    out = pd.DataFrame({"prompt": df[text_col].astype(str)})
    if cat_col:
        out["attack_type"] = df[cat_col].astype(str)
    save(out, os.path.join(UNSAFE_DIR, "jailbreakbench_behaviors.csv"))


def download_wildchat():
    """1M WildChat conversations — extract safe turns + flag toxic ones."""
    print("\n[6/7] WildChat-1M (allenai/WildChat-1M) — capped at 30k")
    try:
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        safe_rows, unsafe_rows = [], []

        for row in ds:
            if len(safe_rows) >= LARGE_DATASET_CAP and len(unsafe_rows) >= 5_000:
                break
            conv = row.get("conversation", [])
            user_turns = [t["content"] for t in conv if t.get("role") == "user"]
            if not user_turns:
                continue
            toxic = row.get("toxic", False) or row.get("redacted", False)
            if toxic and len(unsafe_rows) < 5_000:
                unsafe_rows.append({"prompt": user_turns[0], "attack_type": "wild_toxic"})
            elif not toxic and len(safe_rows) < LARGE_DATASET_CAP:
                safe_rows.append({"prompt": user_turns[0], "label": 0})

        if safe_rows:
            save(pd.DataFrame(safe_rows), os.path.join(SAFE_DIR, "wildchat_safe_30k.csv"))
        if unsafe_rows:
            save(pd.DataFrame(unsafe_rows), os.path.join(UNSAFE_DIR, "wildchat_toxic.csv"))
    except Exception as e:
        print(f"  FAILED: {e}")


def download_hh_rlhf():
    """Anthropic HH-RLHF — human-preferred helpful & harmless conversations."""
    print("\n[7/7] Anthropic HH-RLHF (Anthropic/hh-rlhf)")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    df = ds.to_pandas()

    # Extract the first human turn from the 'chosen' conversation string
    def extract_first_human(text):
        if not isinstance(text, str):
            return None
        for line in text.split("\n"):
            if line.startswith("Human:"):
                return line.replace("Human:", "").strip()
        return None

    df["prompt"] = df["chosen"].apply(extract_first_human)
    df = df[df["prompt"].notna() & (df["prompt"].str.len() > 10)]

    sample = df[["prompt"]].sample(min(len(df), 20_000), random_state=42).assign(label=0)
    save(sample, os.path.join(SAFE_DIR, "hh_rlhf_safe.csv"))


if __name__ == "__main__":
    for fn in [download_beavertails, download_salad, download_categorical_harmful,
               download_lmsys, download_jailbreakbench, download_wildchat, download_hh_rlhf]:
        try:
            fn()
        except Exception as e:
            print(f"  FAILED: {e}")
    print("\nAll done.")
