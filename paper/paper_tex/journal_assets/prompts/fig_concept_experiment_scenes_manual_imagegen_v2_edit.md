Use case: scientific-educational
Task type: edit
Asset type: SCI journal figure for an experiment-scene overview in a computer-science robotics paper
Primary request: Improve the provided experiment scene figure while preserving the three-column layout and scientific-paper style. The figure explains three active-depth sensing failure regimes for a drone passing through a narrow slit: Glare, Specular, and Dark.

Input image role:
- The input image is the current figure to edit. Preserve its overall layout, panel borders, title tabs, labels, color coding, and journal-figure composition.

Technical ground truth from the code implementation:
- Local scene coordinates use x as flight direction and y as lateral direction.
- Start is at x = -1.0, goal is at x = 1.8.
- The front wall is at x = 0.65.
- The physical front wall consists of two vertical wall blocks with a narrow slit between them. There is no physical top or bottom frame across the slit.
- The slit is a vertical opening centered at height z = 1.5 and located at one of four lateral y slots: far_left, left, right, far_right.
- The back wall is a single wall behind the slit, default around x = 2.55, wider in y than the front wall.
- Glare is an opening/backlight effect through the slit.
- Specular and Dark are side-wall patch effects near the two vertical slit edges.

Edit target invariants:
- Preserve the canvas aspect ratio and the three major columns: Glare, Specular, Dark.
- Preserve the top row as a plan-view map, the middle row as third-person scene renderings, and the bottom row as drone-camera / sensor-observation renderings.
- Preserve short labels: "Start", "Goal", "front wall", "back wall", "slit", "far_left", "left", "right", "far_right", "Glare", "Specular", "Dark", "washed-out cue", "false depth", "weak return", and the near/far depth bar.
- Preserve the blue flight/path arrows and dashed sensor-observation boxes, but correct their direction and visual logic.
- Do not introduce unrelated obstacles, people, logos, extra drones, or decorative backgrounds.

Required changes:
1. Top plan-view map:
   - Redraw it so it matches the code geometry: a drone starts at x=-1.0, flies along +x toward a two-block front wall at x=0.65 with a lateral slit/opening in the middle, then toward a single back wall behind the slit and a goal at x=1.8.
   - Show the front wall as two separated horizontal wall blocks in top view with a gap/slit between them; do not show a closed rectangular tunnel or top/bottom doorway frame.
   - Show the back wall as one solid transverse wall behind the slit.
   - Keep the four lateral slot markers far_left, left, right, far_right as dashed vertical sample lines behind the front wall, but make them visually secondary and aligned with y positions.
   - Keep the x/y axis indicator.

2. Middle row, third-person scene panels:
   - The drone must face inward toward the slit in all three panels; the camera lens should point toward the wall opening, not outward toward the viewer.
   - The blue trajectory arrows should point from the drone toward the wall/slit, i.e. into the scene, not toward the drone.
   - Keep the third-person scene readable: walls, slit, back wall, drone, and floor tiles should remain visible.
   - In Glare, show orange/white light at the slit and warm spill on the wall, but do not fully wash out the human-view scene.
   - In Specular, show glossy cyan-white reflections on the side-wall patches near slit edges.
   - In Dark, show dark low-reflectance patches on side walls near the slit edges.

3. Bottom row, drone-camera/sensor observation panels:
   - The slit edges in all three sensor-observation panels should be degraded and not perfectly sharp.
   - Glare observation: show a gray sensor-like view with white washed-out speckle/clouds and bright saturated regions around the slit edges; the slit boundary should be hard to localize.
   - Specular observation: improve it to clearly show false-depth behavior: cyan/white vertical glints near both slit edges, noisy speckles, and curved cyan arrows indicating false near-depth or edge drift. The wall-slit boundary should look unstable, not clean.
   - Dark observation: show weak return with black/dark noisy vertical bands near both slit edges, sparse speckles, and reduced edge contrast.
   - Keep the bottom-row observation images distinct from the middle-row RGB scene views. They should look more like degraded active-depth/sensor outputs.

Style/medium:
- Clean SCI journal figure, semi-photorealistic 3D render panels combined with crisp diagram annotations.
- White background, thin black panel borders, subtle shadows, consistent font scale.
- Keep colors consistent: orange for glare, cyan for specular, dark gray/black for dark, blue for flight path.

Avoid:
- Do not make the top plan-view physically inconsistent with env_cuda.py.
- Do not show arrows pointing toward the drone in the middle row.
- Do not make the drone camera face outward toward the reader.
- Do not make bottom-row slit edges perfectly sharp or clean.
- Do not add too much explanatory text; keep the original compact labels.
