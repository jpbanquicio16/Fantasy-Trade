"""Trade-off chart for the named-player mode (design section 7): one dot per simulated offer, your gain
against the owner's, the bar as a dashed line, and the front (offers no other offer beats for both sides at
once, among those that clear the bar and are not 'unlikely') highlighted. Plain SVG, no plotting dependency.
"""

from __future__ import annotations

from html import escape

W, H, L, R, T, B = 760, 480, 70, 24, 40, 56
GREY, BLUE, RED = "#9aa3ad", "#1f6feb", "#c0392b"


def tradeoff_svg(simulated: list[dict], front: list[dict], bar: float, title: str) -> str:
    """``simulated`` rows carry your_gain, their_pts, tier, label (see targeted.recommend_for_player)."""
    pts = [(s["their_pts"], s["your_gain"]) for s in simulated] or [(0.0, 0.0)]
    xs, ys = [p[0] for p in pts] + [0.0], [p[1] for p in pts] + [0.0, bar]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    px, py = (x1 - x0) * 0.06 or 1.0, (y1 - y0) * 0.08 or 1.0
    x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py
    sx = lambda x: L + (x - x0) / (x1 - x0) * (W - L - R)  # noqa: E731
    sy = lambda y: H - B - (y - y0) / (y1 - y0) * (H - T - B)  # noqa: E731

    def ticks(lo: float, hi: float, n: int = 6) -> list[float]:
        step = (hi - lo) / n
        mag = 10 ** (len(str(int(abs(step)))) - 1) if step >= 1 else 10 ** -(len(str(int(1 / max(step, 1e-9)))))
        step = max(round(step / mag) * mag, mag)
        t, out = (int(lo / step) - 1) * step, []
        while t <= hi:
            if t >= lo:
                out.append(round(t, 6))
            t += step
        return out

    fid = {s["offer_id"] for s in front}
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{L}" y="22" font-size="14" font-weight="600">{escape(title)}</text>']
    for t in ticks(x0, x1):
        o.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{T}" y2="{H - B}" stroke="#e5e8eb"/>'
                 f'<text x="{sx(t):.1f}" y="{H - B + 16}" text-anchor="middle" fill="#555">{t:g}</text>')
    for t in ticks(y0, y1):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="#e5e8eb"/>'
                 f'<text x="{L - 8}" y="{sy(t) + 4:.1f}" text-anchor="end" fill="#555">{t:g}</text>')
    o.append(f'<line x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{T}" y2="{H - B}" stroke="#555"/>')
    o.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(bar):.1f}" y2="{sy(bar):.1f}" stroke="{RED}" stroke-dasharray="5 4"/>'
             f'<text x="{W - R}" y="{sy(bar) - 5:.1f}" text-anchor="end" fill="{RED}">your bar {bar:g}</text>')
    for s in simulated:
        if s["offer_id"] in fid:
            continue
        o.append(f'<circle cx="{sx(s["their_pts"]):.1f}" cy="{sy(s["your_gain"]):.1f}" r="3" fill="{GREY}" '
                 f'fill-opacity="0.6"><title>{escape(s["label"])} ({escape(s["tier"])})</title></circle>')
    for s in front:
        o.append(f'<circle cx="{sx(s["their_pts"]):.1f}" cy="{sy(s["your_gain"]):.1f}" r="5" fill="{BLUE}" '
                 f'stroke="white"><title>{escape(s["label"])} ({escape(s["tier"])})</title></circle>')
    o.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 14}" text-anchor="middle" fill="#333">'
             "owner's change, FP a week (right = better for him)</text>")
    o.append(f'<text transform="translate(16 {(T + H - B) / 2:.0f}) rotate(-90)" text-anchor="middle" fill="#333">'
             "your gain, FP a week</text>")
    o.append(f'<circle cx="{L + 8}" cy="{H - 34}" r="4" fill="{GREY}"/><text x="{L + 18}" y="{H - 30}">simulated offer</text>'
             f'<circle cx="{L + 140}" cy="{H - 34}" r="5" fill="{BLUE}"/><text x="{L + 152}" y="{H - 30}">'
             "on the front: clears your bar, owner not 'unlikely', no offer beats it for both</text>")
    o.append("</svg>")
    return "\n".join(o)
