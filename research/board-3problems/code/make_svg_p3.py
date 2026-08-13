import json

d = json.load(open("p3prof.json"))
u, fw, fa, T = d["u"], d["walk"], d["ar1"], d["T"]
X0, X1, Y0, Y1 = 96.0, 636.0, 62.0, 258.0
YMAX = 0.9


def px(i):
    return X0 + (X1 - X0) * i / (T - 1)


def py(v):
    return Y1 - (Y1 - Y0) * v / YMAX


o = []
A = o.append
H = 382
A(f'<svg width="100%" viewBox="0 0 680 {H}" role="img">')
A('<title>How far back a terminal-frame reward reaches, under two priors</title>')
A('<desc>Line chart of the fractional variance reduction per frame caused by a '
  'quadratic reward on the last of sixteen frames. Under the ring-anchored '
  'random walk the reduction is large everywhere, rising roughly linearly from '
  '0.22 at the first frame to 0.82 at the last. Under the stationary '
  'autoregressive prior it is confined to the final frames, falling by a factor '
  'alpha squared per frame going backwards, from 0.71 at the last frame to '
  '0.0009 at the first, a 251-fold difference at the first frame.</desc>')
A('<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" '
  'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
  '<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" '
  'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
  '</marker></defs>')

A('<text class="th" x="40" y="26">Reward on the last frame: how far back it '
  'reaches</text>')
A('<text class="ts" x="40" y="44">Fractional variance reduction per frame, '
  'T = 16, quadratic reward, s² = 0.4</text>')

for k in range(4):
    v = YMAX * k / 3
    yy = py(v)
    A(f'<line x1="{X0}" y1="{yy:.1f}" x2="{X1}" y2="{yy:.1f}" '
      f'stroke="var(--b)" stroke-width="0.5"/>')
    A(f'<text class="ts" x="{X0 - 10:.0f}" y="{yy + 4:.1f}" '
      f'text-anchor="end">{v:.1f}</text>')
for i in (0, 5, 10, 15):
    A(f'<text class="ts" x="{px(i):.1f}" y="{Y1 + 18:.0f}" '
      f'text-anchor="middle">{i}</text>')
A(f'<text class="ts" x="{(X0 + X1) / 2:.0f}" y="{Y1 + 38:.0f}" '
  'text-anchor="middle">frame index u</text>')

pw = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in zip(u, fw))
pa = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in zip(u, fa))
A(f'<polyline points="{pa}" fill="none" stroke="#D85A30" stroke-width="1.8" '
  'stroke-dasharray="5 3" stroke-linejoin="round"/>')
A(f'<polyline points="{pw}" fill="none" stroke="#7F77DD" stroke-width="1.8" '
  'stroke-linejoin="round"/>')
for i, v in zip(u, fw):
    A(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="2.2" fill="#7F77DD"/>')
for i, v in zip(u, fa):
    A(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="2.2" fill="#D85A30"/>')

ly = 78.0
A(f'<line x1="{X0 + 14}" y1="{ly}" x2="{X0 + 44}" y2="{ly}" '
  'stroke="#7F77DD" stroke-width="1.8"/>')
A(f'<text class="t" x="{X0 + 52}" y="{ly + 4}">ring-anchored walk '
  '(non-stationary)</text>')
A(f'<line x1="{X0 + 14}" y1="{ly + 22}" x2="{X0 + 44}" y2="{ly + 22}" '
  'stroke="#D85A30" stroke-width="1.8" stroke-dasharray="5 3"/>')
A(f'<text class="t" x="{X0 + 52}" y="{ly + 26}">stationary AR(1), '
  'α = 0.8</text>')

yb = Y1 + 52
A(f'<rect class="box" x="40" y="{yb:.0f}" width="600" height="52" rx="4"/>')
A(f'<text class="t" x="56" y="{yb + 21:.0f}">The prior’s memory alone '
  'decides the reach: 0.22 vs 0.0009 at the first frame.</text>')
A(f'<text class="ts" x="56" y="{yb + 39:.0f}">AR(1) decays by exactly '
  'α² per frame going backwards, range ξ = 4.5 frames. The walk '
  'reaches every frame.</text>')
A('</svg>')
open("fig_p3.svg", "w").write("\n".join(o))
print("\n".join(o))
