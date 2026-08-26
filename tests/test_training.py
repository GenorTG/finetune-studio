from finetune_studio.training.engine import TrainingConfig, TrainingState, TrainingEngine

def test_config():
    c = TrainingConfig()
    assert c.lora_rank == 64
    assert c.learning_rate == 8e-5

def test_state():
    s = TrainingState()
    assert s.status == "idle"

def test_engine():
    e = TrainingEngine()
    assert e.state.status == "idle"
