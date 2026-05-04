from epl_prediction_optimizer.storage.results import score_pick


def test_score_pick_uses_three_for_win_one_for_draw_zero_for_loss():
    assert score_pick("Arsenal", "Arsenal", None) == 3
    assert score_pick("Arsenal", None, None) == 1
    assert score_pick("Arsenal", "Everton", None) == 0

