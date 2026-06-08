Use case: scientific-educational
Task type: edit
Asset type: polished SCI journal neural-network architecture figure, 1536x512

Input image role:
The input image contains the correct Split-Stem Policy logic and data flow. Preserve the logic, branch separation, variable meanings, and outputs.

Reference style to imitate:
Use a more expressive neural-network diagram style inspired by a compact transformer/CNN architecture figure: pale pastel modules, thin dark outlines, subtly skewed stacked layer blocks, dotted/hatched header strips, small circular fusion nodes, clean dashed grouping boxes, light internal texture, and elegant technical drawing proportions. The figure should feel more designed and less like plain PowerPoint rectangles.

Primary edit request:
Redraw the same Split-Stem Policy architecture with more visually expressive neural-network elements while keeping the data-flow logic correct. Make each neural network component look like a real module:
- CNN stems should be shown as small stacks of slanted convolution layers, not plain rectangles.
- Gated fusion should be represented by a compact fusion/gate block with two incoming streams and a small gate symbol.
- GRU modules should look like recurrent blocks with a subtle loop/hidden-state motif.
- Camera visual fusion into z_t^c should use a small summation/fusion node receiving F_c direct feature, A_c residual adapter, and G_c spatial adapter.
- The camera branch should remain enclosed by a red dashed boundary labeled "frozen during flight-only".

Preserve exact logic:
- D_obs -> Depth preprocessing -> Phi_t (2 ch., 12x16).
- Phi_t splits to F_f stem and to F_c stem / G_c adapter.
- Flight branch: F_f stem plus s_t -> W_s -> LayerNorm + gated fusion -> GRU_f -> W_u -> u_t.
- W_u is linear. Do not write sigmoid on W_u.
- Camera branch: F_c stem direct feature + A_c residual adapter + G_c adapter -> z_t^c; c_t -> W_c; m_t -> W_m; then concat -> W_p -> GRU_c -> W_o + sigmoid -> c*_{t+1}=[P,E,G].
- Only W_o has sigmoid.

Labels to keep short and readable:
"D_obs", "Depth preprocessing", "Phi_t", "F_f stem", "s_t", "W_s", "Gated fusion", "GRU_f", "W_u", "u_t", "F_c stem", "G_c adapter", "A_c", "z_t^c", "c_t", "m_t", "W_c", "W_m", "concat", "W_p", "GRU_c", "W_o + sigmoid", "c*_{t+1}=[P,E,G]", "frozen during flight-only".

Composition:
1536x512 wide figure. White background. Left-to-right flow. Depth preprocessing on the left, flight branch upper right in pale green, camera branch lower right in pale orange. Keep enough margin. Use large text suitable for a journal figure.

Color and line style:
Light blue for depth preprocessing, pale green for flight branch, pale orange for camera branch, soft purple for output heads, dark gray arrows, muted red dashed camera-freeze outline. Use subtle dotted or grid texture only inside module headers, not in the background.

Avoid:
Changing the data flow, adding new modules, missing c_t or m_t, sigmoid on W_u, tiny unreadable text, decorative drone icons, 3D perspective, heavy shadows, cartoon style, dark background, excessive text, crowded arrows, wrong branch sharing, watermark, logo.
