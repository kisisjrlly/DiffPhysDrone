import torch

from model import Model


def test_gray_model_uses_two_camera_dimensions():
    model = Model(
        dim_obs=10,
        dim_action=3,
        include_camera_state_in_obs=True,
        sensor_type="gray",
        gray_nn_width=16,
        gray_nn_height=12,
    )
    assert model.camera_action_dim == 2
    assert model.cam_state_proj.in_features == 2
    assert model.fc_cam.out_features == 2

    batch = 4
    state = torch.randn(batch, 12)
    gray = torch.rand(batch, 2, 24, 32)
    camera_state = torch.zeros(batch, 2)
    camera_motion = torch.zeros(batch, 6)

    act, cam, hx, cam_hx = model(
        state,
        gray_obs=gray,
        camera_state=camera_state,
        camera_motion_state=camera_motion,
    )
    assert act.shape == (batch, 3)
    assert cam.shape == (batch, 2)
    assert hx.shape == (batch, 192)
    assert cam_hx.shape == (batch, 96)
    assert torch.all((cam >= 0.0) & (cam <= 1.0))


def test_gray_single_frame_is_duplicated_and_resized():
    model = Model(
        sensor_type="gray",
        gray_nn_width=10,
        gray_nn_height=8,
    )
    frame = torch.rand(2, 20, 30)
    x = model.preprocess_gray_input(frame)
    assert x.shape == (2, 2, 8, 10)
    torch.testing.assert_close(x[:, 0], x[:, 1])
    assert float(x.min()) >= -1.0
    assert float(x.max()) <= 1.0


def test_legacy_depth_mode_keeps_three_camera_dimensions():
    model = Model(sensor_type="diff_depth")
    assert model.camera_action_dim == 3
    assert model.cam_state_proj.in_features == 3
    assert model.fc_cam.out_features == 3
