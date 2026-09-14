from utils.runtime_benchmark import (
    cpu_thread_candidates,
    estimate_training_seconds,
    select_best_result,
)


def test_cpu_thread_candidates_cover_small_powers_and_machine_limit():
    assert cpu_thread_candidates(12) == [1, 2, 4, 8, 12]
    assert cpu_thread_candidates(4) == [1, 2, 4]
    assert cpu_thread_candidates(1) == [1]


def test_select_best_result_ignores_failed_candidates():
    rows = [
        {"label": "cpu-4", "status": "ok", "seconds_per_episode": 1.7},
        {"label": "cuda", "status": "ok", "seconds_per_episode": 1.2},
        {"label": "cuda-compile", "status": "error", "seconds_per_episode": None},
    ]
    assert select_best_result(rows)["label"] == "cuda"


def test_select_best_result_keeps_compile_informational_only():
    rows = [
        {"label": "cuda", "status": "ok", "seconds_per_episode": 1.2, "compile_actors": False},
        {"label": "cuda-compile", "status": "ok", "seconds_per_episode": 0.9, "compile_actors": True},
    ]
    assert select_best_result(rows)["label"] == "cuda"


def test_estimate_training_seconds_scales_measured_episode_time():
    assert estimate_training_seconds(1.25, 500) == 625.0
    assert estimate_training_seconds(1.25, 5000) == 6250.0
