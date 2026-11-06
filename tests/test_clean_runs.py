"""Harbor trial names map back to clean problems and pass counts."""

from tagent.data.clean_runs import format_run, problem_id_from_trial, tally_reward_stats


def test_trial_name_keeps_a_compound_index():
    assert problem_id_from_trial("304_B__38422755__3xoSN6G") == "304/B"
    assert problem_id_from_trial("177_A1__106360014__abc") == "177/A1"


def test_tally_counts_passes_and_trials():
    stats = {
        "1.0": ["21_B__1__aa", "21_B__1__bb"],
        "0.0": ["21_B__1__cc", "26_A__2__dd"],
    }
    counts = tally_reward_stats(stats)
    assert counts["21/B"] == {"passes": 2, "trials": 3}
    assert counts["26/A"] == {"passes": 0, "trials": 1}


def test_format_run_says_when_a_model_was_not_exported():
    assert format_run(None) == "not run"
    assert format_run({"passes": 2, "trials": 5, "job": "rest_train"}) == "2/5"
