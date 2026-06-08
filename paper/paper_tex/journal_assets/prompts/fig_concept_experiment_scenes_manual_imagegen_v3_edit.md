Use case: scientific-educational
Task type: edit
Asset type: SCI journal figure correction
Primary request: Apply a targeted correction to the provided edited figure. Keep the improved three-column layout and degraded sensor-observation panels, but fix the top plan-view map and the middle-row blue trajectory arrows.

Edit target invariants:
- Preserve the overall canvas, panel layout, three columns, labels, borders, and improved bottom-row sensor effects from the input image.
- Preserve Glare / Specular / Dark title tabs and bottom dashed boxes.
- Preserve the semi-photorealistic wall/drone/floor style.
- Do not add new explanatory text.

Critical correction 1: top plan-view map must match env_cuda.py.
- Coordinates: x runs left-to-right, y runs vertical on the page.
- Start at x=-1.0 is on the left, goal at x=1.8 is on the right.
- The front wall is located at a fixed x=0.65, so in top view it must appear as a vertical wall line / narrow vertical slab, not horizontal bars.
- The physical front wall consists of exactly two separated vertical wall blocks along y, with a narrow central slit/gap between them. Draw one upper vertical block and one lower vertical block aligned at x=0.65. The gap between them is the slit.
- Do not draw a rectangular tunnel, top/bottom doorway frame, or four little horizontal bars.
- The back wall behind the slit should be a single solid vertical wall slab farther right, around x=2.55.
- Keep dashed slot lines far_left, left, right, far_right behind the front wall as lateral y sample positions. They should be secondary guides, not walls.

Critical correction 2: middle-row trajectory arrows.
- In all three third-person scene panels, the blue dashed flight arrows must point from the drone into the slit, i.e. upward into the scene toward the wall opening.
- Remove any downward arrowhead or double-headed direction. Use repeated arrowheads pointing only toward the wall/slit.
- The drone body and camera lens should visually face the wall/slit, not the viewer. If possible, turn the drone slightly so the front camera is directed into the scene.

Keep from the input:
- Glare middle panel: visible orange light at the slit, but still readable walls.
- Specular middle panel: cyan/white reflective side-wall patches.
- Dark middle panel: dark side-wall patches.
- Bottom Glare: washed-out white speckles and uncertain slit edges.
- Bottom Specular: cyan glints, noisy false-depth speckles, curved cyan arrows, unstable slit boundaries.
- Bottom Dark: black/noisy weak-return bands with sparse speckles.

Avoid:
- Do not degrade the labels into unreadable pseudo-text.
- Do not make the top map physically inconsistent with fixed-x front/back walls.
- Do not keep any blue arrow in the middle row pointing back toward the drone.
