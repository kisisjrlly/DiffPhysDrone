import torch

from model import Model


def test_model_uses_two_camera_dimensions():
    model = Model(
        dim_obs=10,
        dim_action=3,
        include_camera_state_in_obs=True,
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


def test_single_frame_is_duplicated_and_resized():
    model = Model(gray_nn_width=10, gray_nn_height=8)
    frame = torch.rand(2, 20, 30)
    x = model.preprocess_gray_input(frame)
    assert x.shape == (2, 2, 8, 10)
    torch.testing.assert_close(x[:, 0], x[:, 1])
    assert float(x.min()) >= -1.0
    assert float(x.max()) <= 1.0


def test_freeze_modes_select_expected_parameter_groups():
    model = Model()
    frozen_camera = model.freeze_camera_for_flight_only()
    assert frozen_camera
    assert all(
        (not p.requires_grad) if model._is_camera_parameter(name) else p.requires_grad
        for name, p in model.named_parameters()
    )

    model = Model()
    frozen_flight = model.freeze_flight_for_camera_only()
    assert frozen_flight
    assert all(
        p.requires_grad if model._is_camera_parameter(name) else (not p.requires_grad)
        for name, p in model.named_parameters()
    )


def test_camera_visual_warm_start_copies_trained_flight_stem():
    model = Model()
    with torch.no_grad():
        for param in model.stem.parameters():
            param.add_(0.123)
    model.initialize_camera_visual_from_flight()
    for flight, camera in zip(model.stem.parameters(), model.cam_stem.parameters()):
        torch.testing.assert_close(flight, camera)


def test_flight_policy_has_no_direct_camera_state_shortcut_by_default():
    torch.manual_seed(0)
    model = Model(
        dim_obs=10,
        dim_action=3,
        include_camera_state_in_obs=False,
        gray_nn_width=16,
        gray_nn_height=12,
    )
    batch = 3
    state = torch.randn(batch, 10)
    gray = torch.rand(batch, 2, 24, 32)
    motion = torch.zeros(batch, 6)

    act_a, cam_a, _, _ = model(
        state,
        gray_obs=gray,
        camera_state=torch.zeros(batch, 2),
        camera_motion_state=motion,
    )
    act_b, cam_b, _, _ = model(
        state,
        gray_obs=gray,
        camera_state=torch.ones(batch, 2),
        camera_motion_state=motion,
    )

    # Flight can only feel camera actions through future images. The dedicated
    # camera branch is still allowed to condition on its current actuator state.
    torch.testing.assert_close(act_a, act_b)
    assert not torch.allclose(cam_a, cam_b)
