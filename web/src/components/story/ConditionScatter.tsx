// Bklit-style scatter: true means as rings, flow samples as filled dots,
// baseline as open circles. Shape, not just color, tells them apart.

import { PROTOCOL, SCATTER } from '../../data'

const W = 360
const H = 340
const PAD = { l: 36, r: 12, t: 12, b: 32 }
const LIM = 4.2

export function ConditionScatter() {
  const plotW = W - PAD.l - PAD.r
  const plotH = H - PAD.t - PAD.b
  const sx = (v: number) => PAD.l + ((v + LIM) / (2 * LIM)) * plotW
  const sy = (v: number) => PAD.t + plotH - ((v + LIM) / (2 * LIM)) * plotH
  const rTau = (PROTOCOL.tau / (2 * LIM)) * plotW
  const ticks = [-4, 0, 4]

  return (
    <div className="chart-wrap">
      <svg
        className="chart-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Scatter of sampler points and a naive cloud against true target balls. Sampler points sit inside the holdout rings. Naive-cloud points do not."
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} y1={PAD.t} x2={sx(t)} y2={PAD.t + plotH} stroke="var(--chart-grid)" strokeWidth="1" />
            <line x1={PAD.l} y1={sy(t)} x2={PAD.l + plotW} y2={sy(t)} stroke="var(--chart-grid)" strokeWidth="1" />
            <text x={sx(t)} y={H - 8} textAnchor="middle" fontSize="10" fill="var(--chart-label)" fontFamily="var(--font-mono)">
              {t}
            </text>
            <text x={PAD.l - 6} y={sy(t) + 3} textAnchor="end" fontSize="10" fill="var(--chart-label)" fontFamily="var(--font-mono)">
              {t}
            </text>
          </g>
        ))}
        {SCATTER.means.map((m) => (
          <circle
            key={`ring-${m.c}`}
            cx={sx(m.x)}
            cy={sy(m.y)}
            r={rTau}
            fill="none"
            stroke={m.hold ? 'var(--chart-hold)' : 'var(--line-rule)'}
            strokeWidth={m.hold ? 1.8 : 1}
            strokeDasharray={m.hold ? '4 3' : undefined}
          />
        ))}
        {SCATTER.baseline.map((p, i) => (
          <circle
            key={`b${i}`}
            className="mix-cell"
            cx={sx(p.x)}
            cy={sy(p.y)}
            r="3.1"
            fill="none"
            stroke="var(--chart-baseline)"
            strokeWidth="1.3"
            style={{ animationDelay: `${i * 8}ms` }}
          />
        ))}
        {SCATTER.flow.map((p, i) => (
          <circle
            key={`f${i}`}
            className="mix-cell"
            cx={sx(p.x)}
            cy={sy(p.y)}
            r="2.6"
            fill="var(--chart-flow)"
            style={{ animationDelay: `${i * 8 + 80}ms` }}
          />
        ))}
      </svg>
      <ul className="legend">
        <li>
          <span className="swatch" style={{ background: 'var(--chart-flow)', borderColor: 'var(--chart-flow)' }} /> sampler points
        </li>
        <li>
          <span className="swatch" style={{ background: 'transparent', borderColor: 'var(--chart-baseline)' }} /> naive cloud
        </li>
        <li>
          <span className="swatch" style={{ background: 'transparent', borderColor: 'var(--chart-hold)' }} /> holdout target ball
        </li>
      </ul>
    </div>
  )
}
