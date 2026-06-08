Use case: scientific-educational
Task type: edit
Asset type: corrected SCI journal method figure

Edit target invariants:
- Preserve the overall wide 3:1 composition, white background, pastel color palette, clean vector-like style, aligned boxes, and left-to-right flow from the input image.
- Preserve the three regions: depth preprocessing on the left, flight branch on the upper right, camera branch on the lower right.
- Preserve the fact that W_u has no sigmoid and W_o has sigmoid.
- Preserve the separate F_f and F_c stems.
- Preserve G_c as a parallel adapter that receives Phi_t directly.

Primary edit request:
Make the camera visual fusion more accurate to the model and manuscript. In the camera branch, the output of F_c stem should go directly into z_t^c AND also pass through A_c adapter into z_t^c. The G_c adapter should also feed z_t^c. Thus z_t^c should visibly combine three inputs: F_c direct feature, A_c residual adapter, and G_c spatial adapter.

Specific corrections:
- Add or adjust arrows so F_c stem has a direct residual arrow to z_t^c.
- Keep a second arrow from F_c stem to A_c adapter, then from A_c adapter to z_t^c.
- Keep G_c adapter feeding z_t^c in parallel.
- Keep c_t -> W_c and m_t -> W_m as separate inputs to concat.
- If possible, make the final camera output label lowercase: "c*_{t+1} = [P,E,G]".
- Keep labels large and horizontal.
- Do not add any new modules.
- Do not add sigmoid to W_u.
- Do not change the flight branch structure.

Avoid:
Crowding, crossing arrows, extra labels, extra equations, decorative icons, watermarks, misspelled variables, changing the overall style.
