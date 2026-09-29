from types import SimpleNamespace

from llamafactory.train.resume_utils import ResumeMode, get_resume_mode
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer


def _touch(path: str) -> None:
    with open(path, "w", encoding="utf-8"):
        pass


def test_get_resume_mode_auto_uses_model_only_when_rng_world_size_differs(tmp_path):
    checkpoint = tmp_path / "checkpoint-26500"
    checkpoint.mkdir()
    for rank in range(8):
        _touch(str(checkpoint / f"rng_state_{rank}.pth"))

    resume_mode = get_resume_mode(
        str(checkpoint),
        current_world_size=16,
        requested_mode="auto",
        is_deepspeed_enabled=True,
    )

    assert resume_mode.mode == ResumeMode.MODEL_ONLY
    assert resume_mode.saved_world_size == 8
    assert "rng_state files were saved for world size 8" in resume_mode.reason


def test_get_resume_mode_full_keeps_strict_resume_on_world_size_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint-26500"
    checkpoint.mkdir()
    for rank in range(8):
        _touch(str(checkpoint / f"rng_state_{rank}.pth"))

    resume_mode = get_resume_mode(
        str(checkpoint),
        current_world_size=16,
        requested_mode="full",
        is_deepspeed_enabled=True,
    )

    assert resume_mode.mode == ResumeMode.FULL


def test_get_resume_mode_model_only_forces_model_only_without_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint-26500"
    checkpoint.mkdir()
    for rank in range(8):
        _touch(str(checkpoint / f"rng_state_{rank}.pth"))

    resume_mode = get_resume_mode(
        str(checkpoint),
        current_world_size=8,
        requested_mode="model_only",
        is_deepspeed_enabled=True,
    )

    assert resume_mode.mode == ResumeMode.MODEL_ONLY
    assert resume_mode.reason == "resume_from_checkpoint_mode=model_only was requested"


def test_sft_trainer_model_only_resume_loads_weights_then_starts_fresh(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint-26500"
    checkpoint.mkdir()
    for rank in range(8):
        _touch(str(checkpoint / f"rng_state_{rank}.pth"))

    trainer = CustomSeq2SeqTrainer.__new__(CustomSeq2SeqTrainer)
    trainer.args = SimpleNamespace(
        output_dir=str(tmp_path),
        resume_from_checkpoint_mode="auto",
        world_size=16,
    )
    trainer.is_deepspeed_enabled = True
    loaded_checkpoints = []

    def fake_load_from_checkpoint(path):
        loaded_checkpoints.append(path)

    def fake_train(self, resume_from_checkpoint=None, *args, **kwargs):
        return resume_from_checkpoint

    monkeypatch.setattr(trainer, "_load_from_checkpoint", fake_load_from_checkpoint)
    monkeypatch.setattr("transformers.Seq2SeqTrainer.train", fake_train)

    resume_arg = trainer.train(resume_from_checkpoint=str(checkpoint))

    assert loaded_checkpoints == [str(checkpoint)]
    assert resume_arg is None
