import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "Dataset")
SAFE_DIR = os.path.join(DATASET_DIR, "Safe Dataset")
UNSAFE_DIR = os.path.join(DATASET_DIR, "Unsafe Dataset")
WINDOW_SIZE = 5
SAMPLES_PER_CLASS = 150_000  # cap to keep training time reasonable


# ── Attack type normalisation ─────────────────────────────────────────────────
# Collapse raw labels into 6 API categories
ATTACK_TYPE_MAP = {
    # prompt_injection
    "prompt_injection": "prompt_injection",
    "direct_injection": "prompt_injection",
    "indirect_injection": "prompt_injection",
    "payload_injection": "prompt_injection",
    "token_injection": "prompt_injection",
    "indirect injection": "prompt_injection",
    # role_play_manipulation
    "jailbreak": "role_play_manipulation",
    "persona_replacement": "role_play_manipulation",
    "persona_hijacking": "role_play_manipulation",
    "social_engineering": "role_play_manipulation",
    "agent_manipulation": "role_play_manipulation",
    "wild_jailbreak": "role_play_manipulation",
    "wild_toxic": "role_play_manipulation",
    "adversarial": "role_play_manipulation",
    # encoding_obfuscation
    "encoding": "encoding_obfuscation",
    "encoding_obfuscation": "encoding_obfuscation",
    "encoded_attack": "encoding_obfuscation",
    "textfooler": "encoding_obfuscation",
    "bert-attack": "encoding_obfuscation",
    "bae": "encoding_obfuscation",
    "pwws": "encoding_obfuscation",
    "textbugger": "encoding_obfuscation",
    "alzantot": "encoding_obfuscation",
    "pruthi": "encoding_obfuscation",
    "homoglyphs": "encoding_obfuscation",
    "zero width": "encoding_obfuscation",
    "bidirectional text": "encoding_obfuscation",
    "emoji smuggling": "encoding_obfuscation",
    "full width text": "encoding_obfuscation",
    "diacritcs": "encoding_obfuscation",
    "underline accent marks": "encoding_obfuscation",
    "upside down text": "encoding_obfuscation",
    "numbers": "encoding_obfuscation",
    "spaces": "encoding_obfuscation",
    "deletion characters": "encoding_obfuscation",
    "deep word bug": "encoding_obfuscation",
    # incremental_escalation
    "crescendo": "incremental_escalation",
    "multi_turn": "incremental_escalation",
    "multi_turn_drift": "incremental_escalation",
    "many_shot": "incremental_escalation",
    "chain_of_thought": "incremental_escalation",
    "multi turn": "incremental_escalation",
    # system_manipulation
    "system_manipulation": "system_manipulation",
    "instruction_override": "system_manipulation",
    "token_smuggling": "system_manipulation",
    "rag_poisoning": "system_manipulation",
    "context_confusion": "system_manipulation",
    "output_manipulation": "system_manipulation",
    "response_manipulation": "system_manipulation",
    "control": "system_manipulation",
    # data_extraction
    "training_extraction": "data_extraction",
    "prompt_extraction": "data_extraction",
    "prompt_exfiltration": "data_extraction",
    "system_extraction": "data_extraction",
    "prompt_leak": "data_extraction",
    "model_fingerprinting": "data_extraction",
    "code_execution": "data_extraction",
}

def normalise_attack_type(raw):
    if not isinstance(raw, str):
        return "unknown"
    key = raw.strip().lower()
    return ATTACK_TYPE_MAP.get(key, "unknown")


# ── Safe loaders ──────────────────────────────────────────────────────────────

def load_safe():
    texts = []

    # Original research datasets
    for filename in ["alpaca_52k.csv", "dolly_15k.csv", "oasst1_human_turns.csv",
                     "self_instruct.csv", "sharegpt_20k.csv", "ultrachat_30k.csv"]:
        path = os.path.join(SAFE_DIR, filename)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, encoding="utf-8")
        col = "prompt" if "prompt" in df.columns else df.columns[0]
        batch = [t for t in df[col].dropna().astype(str).tolist() if len(t) > 10]
        texts.extend(batch)
        print(f"  {filename}: {len(batch)}")

    # New datasets
    for filename, col in [("beavertails_safe.csv", "prompt"),
                           ("wildchat_safe_30k.csv", "prompt"),
                           ("hh_rlhf_safe.csv", "prompt")]:
        path = os.path.join(SAFE_DIR, filename)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, encoding="utf-8")
        col_name = col if col in df.columns else df.columns[0]
        batch = [t for t in df[col_name].dropna().astype(str).tolist() if len(t) > 10]
        texts.extend(batch)
        print(f"  {filename}: {len(batch)}")

    return texts


# ── Unsafe loaders ────────────────────────────────────────────────────────────

def load_unsafe():
    """Returns list of (text, attack_type) tuples."""
    samples = []

    def add(text_series, type_series=None, default_type="unknown"):
        for i, t in enumerate(text_series.dropna().astype(str)):
            if len(t) < 5:
                continue
            atype = type_series.iloc[i] if type_series is not None else default_type
            samples.append((t, normalise_attack_type(str(atype))))

    # ── Original research datasets ──
    df = pd.read_csv(os.path.join(UNSAFE_DIR, "hlyn_injection_judge_50k.csv"), encoding="utf-8")
    df = df[df["label"] == 1]
    add(df["text"], default_type="prompt_injection")
    print(f"  hlyn: {len(df)}")

    df = pd.read_csv(os.path.join(UNSAFE_DIR, "necent_jailbreak_aggregated.csv"), encoding="utf-8")
    if "prompt_harmful" in df.columns:
        df = df[df["prompt_harmful"] == 1]
    add(df["prompt"], df["attack_technique"] if "attack_technique" in df.columns else None)
    print(f"  necent: {len(df)}")

    df = pd.read_csv(os.path.join(UNSAFE_DIR, "mindgard_evaded_injection.csv"), encoding="utf-8")
    add(df["modified_sample"], df["attack_name"] if "attack_name" in df.columns else None)
    print(f"  mindgard: {len(df)}")

    df = pd.read_csv(os.path.join(UNSAFE_DIR, "neuralchemy_prompt_injection.csv"), encoding="utf-8")
    if "label" in df.columns:
        df = df[df["label"] == 1]
    add(df["text"], df["category"] if "category" in df.columns else None)
    print(f"  neuralchemy: {len(df)}")

    df = pd.read_csv(os.path.join(UNSAFE_DIR, "codesagar_malicious_prompts.csv"), encoding="utf-8")
    if "gpt-malicious" in df.columns:
        df = df[df["gpt-malicious"] == True]
    add(df["prompt"], default_type="malicious_code")
    print(f"  codesagar: {len(df)}")

    df = pd.read_csv(os.path.join(UNSAFE_DIR, "promptwall_injection.csv"), encoding="utf-8")
    df = df[df["label"].isna() | (df["label"] != "safe")]
    add(df["prompt"], df["attack_type"] if "attack_type" in df.columns else None)
    print(f"  promptwall: {len(df)}")

    df = pd.read_csv(os.path.join(UNSAFE_DIR, "trustairlab_jailbreak_wild.csv"), encoding="utf-8")
    add(df["prompt"], default_type="wild_jailbreak")
    print(f"  trustair: {len(df)}")

    # ── New datasets ──
    df = pd.read_csv(os.path.join(UNSAFE_DIR, "beavertails_unsafe.csv"), encoding="utf-8")
    add(df["prompt"], df["attack_type"] if "attack_type" in df.columns else None)
    print(f"  beavertails_unsafe: {len(df)}")

    df = pd.read_csv(os.path.join(UNSAFE_DIR, "salad_attack.csv"), encoding="utf-8")
    add(df["prompt"], df["attack_type"] if "attack_type" in df.columns else None)
    print(f"  salad: {len(df)}")

    df = pd.read_csv(os.path.join(UNSAFE_DIR, "categorical_harmful_qa.csv"), encoding="utf-8")
    text_col = next((c for c in ["Question", "question", "prompt"] if c in df.columns), df.columns[0])
    cat_col = next((c for c in ["Category", "category"] if c in df.columns), None)
    add(df[text_col], df[cat_col] if cat_col else None)
    print(f"  categorical_harmful_qa: {len(df)}")

    df = pd.read_csv(os.path.join(UNSAFE_DIR, "jailbreakbench_behaviors.csv"), encoding="utf-8")
    text_col = next((c for c in ["Goal", "Behavior", "prompt"] if c in df.columns), df.columns[0])
    cat_col = next((c for c in ["Category", "category"] if c in df.columns), None)
    add(df[text_col], df[cat_col] if cat_col else None)
    print(f"  jailbreakbench: {len(df)}")

    df = pd.read_csv(os.path.join(UNSAFE_DIR, "wildchat_toxic.csv"), encoding="utf-8")
    add(df["prompt"], df["attack_type"] if "attack_type" in df.columns else None)
    print(f"  wildchat_toxic: {len(df)}")

    return samples


def build_windows(texts, window_size=5, n_windows=None):
    """
    Concatenate N random independent prompts into one window string.
    Format: USER: t1 [SEP] ASST: t2 [SEP] ... USER: tN
    """
    roles = ["USER", "ASST"]
    if n_windows is None:
        n_windows = len(texts)
    windows = []
    while len(windows) < n_windows:
        sample = random.sample(texts, min(window_size, len(texts)))
        parts = [f"{roles[i % 2]}: {t}" for i, t in enumerate(sample)]
        windows.append(" [SEP] ".join(parts))
    return windows[:n_windows]


if __name__ == "__main__":
    print("=== Loading safe data ===")
    safe_texts = load_safe()
    print(f"Total safe: {len(safe_texts)}\n")

    print("=== Loading unsafe data ===")
    unsafe_samples = load_unsafe()
    unsafe_texts = [t for t, _ in unsafe_samples]
    unsafe_types = [a for _, a in unsafe_samples]
    print(f"Total unsafe: {len(unsafe_samples)}\n")

    # Cap and balance
    n = min(len(safe_texts), len(unsafe_texts), SAMPLES_PER_CLASS)
    print(f"Using {n} samples per class ({n*2} total) ...")

    random.shuffle(safe_texts)
    safe_sample = safe_texts[:n]

    unsafe_idx = random.sample(range(len(unsafe_texts)), n)
    unsafe_sample = [unsafe_texts[i] for i in unsafe_idx]
    unsafe_type_sample = [unsafe_types[i] for i in unsafe_idx]

    print(f"Building windows (N={WINDOW_SIZE})...")
    safe_windows = build_windows(safe_sample, window_size=WINDOW_SIZE, n_windows=n)
    unsafe_windows = build_windows(unsafe_sample, window_size=WINDOW_SIZE, n_windows=n)

    df = pd.DataFrame({
        "text": safe_windows + unsafe_windows,
        "label": [0] * n + [1] * n,
        "attack_type": ["safe"] * n + unsafe_type_sample,
    }).sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    out_dir = os.path.dirname(__file__)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False, encoding="utf-8")

    print(f"\nDone. Train: {len(train_df)}, Test: {len(test_df)}")
    print("\nAttack type distribution (unsafe only):")
    print(df[df["label"] == 1]["attack_type"].value_counts().to_string())
