"""A diagnostic or failed preparation must not become a learned-success movie."""
import json

from tools.render_joint_slit_video import render


def test_no_joint_candidate_does_not_generate_a_success_video(tmp_path):
    (tmp_path/'final_results.json').write_text(json.dumps(dict(joint_candidate=False)))
    render(tmp_path)
    status = json.loads((tmp_path/'demo_status.json').read_text())
    assert status['produced'] is False
    assert not list(tmp_path.rglob('*.mp4'))


def test_failed_common_flight_is_disclosed_without_scripted_flight_substitution(tmp_path):
    (tmp_path/'decision.json').write_text(json.dumps(dict(category='common_flight_not_established')))
    render(tmp_path)
    status = json.loads((tmp_path/'demo_status.json').read_text())
    assert status['produced'] is False
    assert 'not started' in status['reason']
