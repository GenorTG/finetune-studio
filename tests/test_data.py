import json, tempfile, os
from finetune_studio.training.data import load_jsonl, validate_messages, format_for_sft

def test_load_jsonl():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        f.write(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
        path = f.name
    data = load_jsonl(path)
    assert len(data) == 1
    os.unlink(path)

def test_validate_good():
    data = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}]
    assert len(validate_messages(data)) == 0

def test_validate_bad():
    data = [{"messages": [{"content": "hi"}]}]
    assert len(validate_messages(data)) > 0

def test_format_sft():
    data = [{"messages": [{"role": "user", "content": "hi"}]}]
    result = format_for_sft(data, system_prompt="Be helpful.")
    assert result[0]["messages"][0]["role"] == "system"
