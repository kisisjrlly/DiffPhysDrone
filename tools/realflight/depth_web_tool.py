#!/usr/bin/env python3
"""Live D455 depth viewer + real-time camera parameter control (web UI).

Runs on the onboard computer (ROS1 Noetic). The host only needs a browser:

    http://<onboard-ip>:8090

The page shows the colorized depth stream and sliders for exposure / gain /
laser power / emitter / auto exposure. Every slider change is applied through
the realsense2_camera dynamic_reconfigure service, so the effect is visible
immediately in the depth image.
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import rospy
from dynamic_reconfigure.msg import BoolParameter, Config, DoubleParameter, IntParameter
from dynamic_reconfigure.srv import ReconfigureRequest
from sensor_msgs.msg import Image


STATE_LOCK = threading.Lock()
STATE = {
    "depth": None,          # latest sensor_msgs/Image
    "depth_stamp": 0.0,
    "params": {},           # name -> value from parameter_updates
    "ranges": {},           # name -> {type, min, max, def}
}

SET_PROXY = None


def depth_to_jpeg(msg):
    if msg is None:
        return None
    try:
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    except Exception:
        return None
    meters = depth.astype(np.float32) * 0.001
    valid = depth > 0
    norm = np.clip((meters - 0.2) / (5.0 - 0.2), 0.0, 1.0)
    img8 = (norm * 255.0).astype(np.uint8)
    img8[~valid] = 0
    color = cv2.applyColorMap(img8, cv2.COLORMAP_JET)

    with STATE_LOCK:
        p = dict(STATE["params"])
    text = "exposure=%s gain=%s laser=%s emitter=%s ae=%s" % (
        p.get("exposure", "-"),
        p.get("gain", "-"),
        p.get("laser_power", "-"),
        p.get("emitter_enabled", "-"),
        p.get("enable_auto_exposure", "-"),
    )
    cv2.putText(color, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return buf.tobytes() if ok else None


def on_depth(msg):
    with STATE_LOCK:
        STATE["depth"] = msg
        STATE["depth_stamp"] = time.time()


def on_updates(msg):
    with STATE_LOCK:
        for p in msg.bools:
            STATE["params"][p.name] = bool(p.value)
        for p in msg.ints:
            STATE["params"][p.name] = int(p.value)
        for p in msg.doubles:
            STATE["params"][p.name] = float(p.value)


def on_descriptions(msg):
    with STATE_LOCK:
        mins = {p.name: p.value for p in msg.min.ints}
        mins.update({p.name: p.value for p in msg.min.doubles})
        maxs = {p.name: p.value for p in msg.max.ints}
        maxs.update({p.name: p.value for p in msg.max.doubles})
        dflts = {p.name: p.value for p in msg.dflt.ints}
        dflts.update({p.name: p.value for p in msg.dflt.doubles})
        for group in msg.groups:
            for p in group.parameters:
                STATE["ranges"][p.name] = {
                    "type": p.type,
                    "min": mins.get(p.name),
                    "max": maxs.get(p.name),
                    "def": dflts.get(p.name),
                }


def set_param(name, raw_value):
    with STATE_LOCK:
        rng = STATE["ranges"].get(name)
    ptype = rng["type"] if rng else "int"
    try:
        if ptype == "bool" or name == "enable_auto_exposure":
            value = raw_value.lower() in ("1", "true", "on", "yes")
        elif ptype == "double":
            value = float(raw_value)
        else:
            value = int(float(raw_value))
    except ValueError:
        return {"ok": False, "error": "bad value: %r" % raw_value}

    req = ReconfigureRequest()
    req.config = Config()
    if ptype == "bool" or name == "enable_auto_exposure":
        req.config.bools = [BoolParameter(name, value)]
    elif ptype == "double":
        req.config.doubles = [DoubleParameter(name, value)]
    else:
        req.config.ints = [IntParameter(name, value)]
    try:
        SET_PROXY.call(req)
        return {"ok": True, "name": name, "value": value}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def state_json():
    with STATE_LOCK:
        params = dict(STATE["params"])
        ranges = dict(STATE["ranges"])
        last = STATE["depth_stamp"]
        has_depth = STATE["depth"] is not None
    fps = 0.0
    if last and (time.time() - last) < 3.0:
        fps = 15.0  # display only; topic is 15 Hz by default
    return {
        "params": params,
        "ranges": ranges,
        "fps": fps,
        "has_depth": has_depth,
    }


PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>D455 深度监控</title>
<style>
  body { background:#111; color:#ddd; font-family: sans-serif; margin:16px; }
  h1 { font-size:18px; }
  #stream { width:100%; max-width:960px; background:#000; border:1px solid #333; }
  .row { margin:10px 0; max-width:960px; }
  label { display:inline-block; width:180px; }
  input[type=range] { width:480px; vertical-align:middle; }
  .val { display:inline-block; width:90px; text-align:right; color:#8cf; }
  #status { color:#9d9; font-size:13px; }
  .err { color:#f77; }
</style>
</head>
<body>
<h1>D455 深度实时监控（机载 @@HOST@@）</h1>
<img id="stream" src="/depth.mjpeg">
<div class="row" id="controls"></div>
<div id="status">连接中…</div>
<script>
let controls = @@CONTROLS@@;
let timers = {};
function setParam(name, value) {
  fetch('/set?name=' + encodeURIComponent(name) + '&value=' + encodeURIComponent(value))
    .then(r => r.json())
    .then(j => {
      const s = document.getElementById('status');
      if (j.ok) s.textContent = name + ' = ' + j.value + ' 已生效';
      else { s.textContent = name + ' 设置失败: ' + j.error; s.className = 'err'; }
    });
}
function debounce(name, value, wait) {
  clearTimeout(timers[name]);
  timers[name] = setTimeout(() => setParam(name, value), wait);
}
function render() {
  const box = document.getElementById('controls');
  box.innerHTML = '';
  for (const c of controls) {
    const row = document.createElement('div');
    row.className = 'row';
    const label = document.createElement('label');
    label.textContent = c.label;
    row.appendChild(label);
    if (c.type === 'bool') {
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.dataset.name = c.name;
      cb.checked = !!c.value;
      cb.onchange = () => setParam(c.name, cb.checked ? 1 : 0);
      row.appendChild(cb);
    } else if (c.type === 'select') {
      const sel = document.createElement('select');
      sel.dataset.name = c.name;
      for (const [v, t] of c.options) {
        const opt = document.createElement('option');
        opt.value = v; opt.textContent = t;
        if (String(v) === String(c.value)) opt.selected = true;
        sel.appendChild(opt);
      }
      sel.onchange = () => setParam(c.name, sel.value);
      row.appendChild(sel);
    } else {
      const slider = document.createElement('input');
      slider.type = 'range';
      slider.dataset.name = c.name;
      slider.min = c.min; slider.max = c.max; slider.step = c.step;
      slider.value = c.value;
      const val = document.createElement('span');
      val.className = 'val'; val.textContent = c.value;
      slider.oninput = () => {
        val.textContent = slider.value;
        debounce(c.name, slider.value, 120);
      };
      row.appendChild(slider); row.appendChild(val);
    }
    box.appendChild(row);
  }
}
async function refresh() {
  try {
    const s = await (await fetch('/state')).json();
    const status = document.getElementById('status');
    status.textContent = (s.has_depth ? '深度流正常' : '等待深度话题…') +
      ' | 当前参数已同步';
    status.className = '';
    for (const c of controls) {
      if (s.params[c.name] === undefined) continue;
      c.value = s.params[c.name];
      if (c.type === 'bool') {
        const cb = document.querySelector('[data-name="' + c.name + '"]');
        if (cb && document.activeElement !== cb) cb.checked = !!c.value;
      } else if (c.type === 'select') {
        const sel = document.querySelector('[data-name="' + c.name + '"]');
        if (sel && document.activeElement !== sel) sel.value = String(c.value);
      } else {
        const slider = document.querySelector('[data-name="' + c.name + '"]');
        if (slider && document.activeElement !== slider) slider.value = c.value;
        const val = slider ? slider.nextElementSibling : null;
        if (val && (!slider || document.activeElement !== slider)) {
          val.textContent = c.value;
        }
      }
    }
  } catch (e) { /* keep old controls */ }
}
render();
setInterval(refresh, 1500);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep console quiet
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            try:
                with STATE_LOCK:
                    controls = build_controls(STATE["params"], STATE["ranges"])
                body = PAGE.replace("@@HOST@@", args.host).replace(
                    "@@CONTROLS@@", json.dumps(controls))
                self._send(200, body.encode("utf-8"))
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                self._send(500, str(e).encode("utf-8"))
        elif path == "/state":
            self._send(200, json.dumps(state_json()).encode("utf-8"),
                       "application/json")
        elif path == "/depth.mjpeg":
            self._serve_mjpeg()
        elif path == "/set":
            qs = parse_qs(parsed.query)
            name = qs.get("name", [""])[0]
            value = qs.get("value", [""])[0]
            result = set_param(name, value)
            self._send(200, json.dumps(result).encode("utf-8"),
                       "application/json")
        else:
            self._send(404, b"not found")

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        last = None
        while True:
            with STATE_LOCK:
                msg = STATE["depth"]
            jpeg = depth_to_jpeg(msg)
            if jpeg is not None and jpeg != last:
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(b"Content-Length: %d\r\n\r\n" % len(jpeg))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    last = jpeg
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
            time.sleep(0.05)


def build_controls(params, ranges):
    def r(name, default):
        rng = ranges.get(name, {})
        return {
            "min": rng.get("min", default[0]),
            "max": rng.get("max", default[1]),
            "step": default[2],
        }
    ae_rng = r("exposure", (1, 200000, 50))
    gain_rng = r("gain", (16, 248, 1))
    laser_rng = r("laser_power", (0, 360, 1))
    return [
        {"name": "enable_auto_exposure", "label": "自动曝光", "type": "bool",
         "value": params.get("enable_auto_exposure", False)},
        {"name": "exposure", "label": "曝光 (us)", "type": "range",
         "value": params.get("exposure", 8500), **ae_rng},
        {"name": "gain", "label": "增益", "type": "range",
         "value": params.get("gain", 16), **gain_rng},
        {"name": "laser_power", "label": "激光功率 (mW)", "type": "range",
         "value": params.get("laser_power", 150), **laser_rng},
        {"name": "emitter_enabled", "label": "发射器", "type": "select",
         "value": params.get("emitter_enabled", 0),
         "options": [[0, "关闭"], [1, "激光"], [2, "激光自动"], [3, "LED"]]},
    ]


def main():
    global args, SET_PROXY
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--topic", default="/camera/depth/image_rect_raw")
    args = parser.parse_args()

    rospy.init_node("depth_web_tool", anonymous=True)
    rospy.Subscriber(args.topic, Image, on_depth, queue_size=1)
    rospy.Subscriber("/camera/stereo_module/parameter_updates", Config, on_updates)
    rospy.Subscriber("/camera/stereo_module/parameter_descriptions",
                     __import__("dynamic_reconfigure.msg", fromlist=["ConfigDescription"]).ConfigDescription,
                     on_descriptions)
    SET_PROXY = rospy.ServiceProxy("/camera/stereo_module/set_parameters",
                                   __import__("dynamic_reconfigure.srv", fromlist=["Reconfigure"]).Reconfigure)
    threading.Thread(target=rospy.spin, daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("depth web tool listening on http://%s:%d" % (args.host, args.port))
    server.serve_forever()


if __name__ == "__main__":
    main()
