# Hardware Plan — IMX900 Grayscale Camera on the Current Drone

## 1. Current drone hardware

### Compute

- NVIDIA Jetson Orin NX 8GB module.
- Carrier: **DAMIAO / 达妙科技 DM-ORIN NX V2.X**.
- Carrier is a lightweight board intended for compact robotics/UAV systems.
- Carrier manual states:
  - weight: 39.2 g;
  - input: 12–28 V;
  - supports 6S power;
  - size: 50 × 86.8 × 13 mm;
  - three USB 3.0 Type-C interfaces;
  - two 22-pin 0.5-mm FPC CIS connectors;
  - Orin NX and Orin Nano compatibility.

### Flight controller

- NxtPX4 v2;
- H7 MCU;
- dual BMI088 IMUs;
- Jetson flight link via UART/MAVLink;
- development/QGC link kept separate via USB.

### Airframe / propulsion

- XI35 3.5-inch frame;
- ~200 mm motor-to-motor diagonal;
- 2006-class 2006–2150 KV motors;
- D90S ~3.5-inch propellers;
- 6S 1300 mAh 95C battery;
- prop guards currently documented.

### Legacy sensor

- Intel RealSense D455.
- The D455 is legacy/reference hardware for this new branch.
- The planned final grayscale method should not depend on D455 depth.

## 2. Selected new camera

**e-con Systems e-CAM37M_CUONX, monochrome, Sony IMX900.**

Reason for selection:

- global shutter;
- monochrome output avoids RGB ISP/white-balance complexity;
- MIPI CSI-2 eliminates a bulky USB camera/cable in the final configuration;
- e-con targets Jetson Orin NX/Nano;
- exposure and gain are intended to be software-controllable;
- suitable for calibrating a direct simulator-to-hardware action mapping.

## 3. Carrier CSI interfaces

The DAMIAO manual documents two 22-pin 0.5-mm FPC camera connectors: CIS0 and CIS1.

CIS0 exposes two 2-lane CSI groups plus:

- CAM0_PWDN_LS;
- CAM0_MCLK;
- CAM0_I2C_SCL/SDA;
- 3.3 V and GND.

CIS1 similarly exposes CSI2/CSI3 plus:

- CAM1_PWDN_LS;
- CAM1_MCLK;
- CAM1_I2C_SCL/SDA;
- 3.3 V and GND.

The manual explicitly notes that this camera circuitry follows the original/reference arrangement. This makes compatibility promising, but it does **not** prove that e-con's NVIDIA-reference device-tree package will boot unchanged on this third-party carrier.

## 4. Hardware compatibility gate — must be completed before relying on MIPI

Before implementation is considered deployable, verify all of the following.

### Gate A — connector/mechanical pinout

Confirm with DAMIAO and e-con:

- 22-pin connector orientation;
- pin numbering;
- cable contact side;
- lane mapping;
- clock lanes;
- lane polarity;
- 3.3 V behavior;
- I2C levels;
- MCLK;
- PWDN/reset behavior.

Do not connect the camera based only on “22-pin 0.5 mm” matching.

### Gate B — CSI lane topology

Obtain the exact e-con driver/device-tree settings:

- `num_lanes`;
- `tegra_sinterface`;
- CSI port;
- VI channel;
- lane polarity;
- pixel format;
- clock parameters.

Map them to the DAMIAO CIS0/CIS1 wiring.

### Gate C — BSP/JetPack

Record the deployed system:

~~~bash
cat /etc/nv_tegra_release
uname -a
cat /proc/device-tree/model
tr '\\0' '\\n' < /proc/device-tree/compatible
~~~

Then confirm that e-con supplies a compatible package or will port it.

### Gate D — camera control

After the camera streams, enumerate controls:

~~~bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/videoX --list-formats-ext
v4l2-ctl -d /dev/videoX --list-ctrls
v4l2-ctl -d /dev/videoX --all
~~~

Capture exact names/ranges/defaults for:

- exposure;
- gain;
- frame rate;
- trigger mode if present;
- output pixel format;
- any auto-exposure/auto-gain control.

The project simulator must use these real ranges/mappings, not guessed values.

## 5. Questions to send to DAMIAO

Recommended wording:

> We use the DM-ORIN NX V2.X carrier with a Jetson Orin NX 8GB. We plan to connect an e-con Systems e-CAM37M_CUONX (Sony IMX900 monochrome) MIPI CSI-2 camera. Please confirm whether CIS0/CIS1 are electrically and pin-for-pin compatible with the NVIDIA Orin Nano Developer Kit P3768 22-pin camera connectors, including CSI lane mapping/polarity, CAM I2C, MCLK, PWDN/reset, voltage rails and cable orientation. Does a camera device-tree overlay written for P3768 work unchanged, or are carrier-specific DT changes required?

## 6. Questions to send to e-con Systems

Recommended wording:

> We use a Jetson Orin NX 8GB on a DAMIAO DM-ORIN NX V2.X carrier. The carrier has two 22-pin 0.5-mm MIPI CSI connectors and its manual states that the camera circuitry follows the NVIDIA reference/original arrangement. We plan to use e-CAM37M_CUONX with the monochrome Sony IMX900. Please confirm mechanical/electrical compatibility and whether your current Jetson driver/BSP package supports this carrier. If not, can you provide the device-tree changes or a porting service? We also need runtime manual control of exposure time and gain with auto controls disabled, and we need to know command-to-frame latency / whether effective settings are reported per frame.

## 7. Real-time control requirements

The final runtime camera wrapper should expose:

~~~python
set_exposure_us(value)
set_gain(value)
get_requested_settings()
get_effective_settings_if_available()
grab_frame_with_timestamp()
~~~

Every recorded frame should log:

- frame timestamp;
- sequence number;
- requested exposure/gain;
- effective exposure/gain if metadata exposes them;
- command timestamp;
- camera device timestamp if available;
- Jetson receive timestamp.

This is required to measure actuation latency and align the simulator.

## 8. D455 transition strategy

During bring-up only, D455 may remain mounted or used on a bench for:

- safety/reference depth;
- trajectory debugging;
- independent collision validation.

For main experimental claims:

- grayscale policy input must not contain D455 depth;
- camera policy must not contain D455-derived privileged information;
- D455 may be removed to reduce weight once the grayscale system is reliable.

## 9. Update this file after hardware arrival

Record:

- exact e-con SKU;
- lens SKU/focal length/FOV;
- cable SKU and length;
- JetPack/L4T version;
- e-con driver version;
- `v4l2-ctl --all` output;
- tested formats/FPS;
- measured end-to-end latency;
- measured exposure/gain update latency;
- camera + lens + cable mass;
- measured power draw.
