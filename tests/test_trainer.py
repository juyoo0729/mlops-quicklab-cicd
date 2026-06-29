import trainer


def test_config():
    assert trainer.NUM_LABELS == 2
    assert trainer.MODEL_NAME == "klue/bert-base"


def test_functions_exist():
    assert hasattr(trainer, "train")
    assert hasattr(trainer, "evaluate")
    assert hasattr(trainer, "save_model")
    assert hasattr(trainer, "load_data")
