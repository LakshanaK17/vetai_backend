"""Optional LLM decision layer. Loaded only when USE_LLM=1.
Strictly grounded on the retrieved BSAVA rule; never changes the drug."""
import os

_tok = None
_llm = None
_is_seq2seq = False
_device = "cpu"

SYSTEM = ("You are a veterinary decision-support assistant. Obey the retrieved BSAVA rule: "
          "keep EXACTLY the recommended agent/drug it specifies and never substitute a different "
          "drug. Only expand it into practical guidance. The administration route MUST match the "
          "agent: a topical or otic agent is applied to the skin/ear (never given orally); a "
          "systemic agent is given orally or by injection. Always state that a licensed veterinarian "
          "must confirm dosage, contraindications and severity before use. Be concise and clinical.")

def load(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Load the model once at startup."""
    global _tok, _llm, _is_seq2seq, _device
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    _is_seq2seq = any(k in model_name.lower() for k in ["t5", "bart", "flan"])
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tok = AutoTokenizer.from_pretrained(model_name)
    if _is_seq2seq:
        _llm = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(_device)
    else:
        _llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(_device)
    _llm.eval()
    print(f"[llm_layer] loaded {model_name} on {_device}")

def _build_prompt(rec: dict):
    r, d = rec["rule"], rec["diet"]
    facts = (
        f"PATIENT: breed = {rec['breed']}, detected lesion = {rec['lesion']} "
        f"(rule category = {rec['category']}).\n"
        f"RETRIEVED BSAVA RULE (authoritative, do not change the agent):\n"
        f"  - Recommended agent: {r['agent']}\n"
        f"  - Source guideline: {r['source']}\n"
        f"  - Rule trigger: {r['trigger']}\n"
        f"BREED DIET FACTS: profile = {d['profile']}; recommended = {', '.join(d['recommended'])}; "
        f"avoid = {', '.join(d['avoid'])}; condition tip = {rec['diet_modifier']}.\n"
    )
    task = (
        "Using ONLY the facts above, write the recommendation in three short sections:\n"
        "1) TREATMENT - restate the retrieved agent verbatim, then add administration route "
        "(matching the agent), monitoring and breed-specific cautions. Do NOT invent a different drug.\n"
        "2) DIET - a breed-aware plan from the diet facts (ingredients, rough daily portioning, foods to avoid).\n"
        "3) SAFETY - one line that a licensed vet must confirm dosage, contraindications and severity.\n"
    )
    return facts, task

def generate(rec: dict, max_new_tokens: int = 320) -> str:
    import torch
    facts, task = _build_prompt(rec)
    if _is_seq2seq:
        prompt = SYSTEM + "\n\n" + facts + "\n" + task
        ids = _tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(_device)
        with torch.no_grad():
            out = _llm.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        return _tok.decode(out[0], skip_special_tokens=True).strip()
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": facts + "\n" + task}]
    prompt = _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = _tok(prompt, return_tensors="pt").to(_device)
    with torch.no_grad():
        out = _llm.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=_tok.eos_token_id)
    return _tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
