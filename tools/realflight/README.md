# Real-flight Camera Integration — e-con IMX900

The legacy RealSense D455 web/depth control helpers have been removed from this
branch. The selected primary camera is:

- e-con Systems e-CAM37M_CUONX
- Sony IMX900 monochrome global shutter
- MIPI CSI-2
- Jetson Orin NX 8GB on DAMIAO DM-ORIN NX V2.X carrier

## Current status

The real IMX900 runtime node is not implemented yet because the exact e-con
driver/BSP, V4L2 control names/ranges, pixel formats, and command-to-effective-
frame latency must be measured on the purchased hardware first.

See:

- `docs/grayscale_imx900/HARDWARE_IMX900.md`
- `docs/grayscale_imx900/CALIBRATION_SIM2REAL.md`

## Planned runtime interface

The future wrapper should provide explicit operations equivalent to:

```python
set_exposure_us(value)
set_gain(value)
grab_frame_with_timestamp()
get_requested_settings()
get_effective_settings_if_available()
```

Auto exposure/auto gain must be disabled for the main experiments.

## Generic remote helpers

`remote.py` and `upload.py` are retained because they are generic SSH
utilities and are not tied to D455.
