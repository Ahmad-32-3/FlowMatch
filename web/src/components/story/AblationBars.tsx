import { ABLATION, METRICS } from '../../data'

const W = 420
const H = 28 + ABLATION.length * 28
const LEFT = 96
const RIGHT = 52
const BAR = W - LEFT - RIGHT

export function AblationBars() {
  const max = Math.max(100, METRICS.successPct, ...ABLATION.map((r) => r.successPct))
  const x = (v: number) => LEFT + (v / max) * BAR

  return (
    <div className="chart-wrap">
      <svg
        className="chart-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Ablation bars. Sampler hit rate stays high across step counts and widths. A one-number condition encoding is weaker. The naive cloud bar stays near zero."
      >
        {ABLATION.map((r, i) => {
          const y = 8 + i * 28
          return (
            <g key={r.tag}>
              <text x={LEFT - 8} y={y + 11} textAnchor="end" fontSize="11" fill="var(--chart-label)" fontFamily="var(--font-mono)">
                {r.tag}
              </text>
              <rect x={LEFT} y={y + 4} width={BAR} height="12" fill="var(--bg-sunk)" />
              <rect
                className="bar-grow"
                x={LEFT}
                y={y + 4}
                width={Math.max(2, x(r.successPct) - LEFT)}
                height="12"
                fill="var(--chart-flow)"
                style={{ animationDelay: `${i * 40}ms` }}
              />
              <rect
                x={LEFT}
                y={y + 4}
                width={Math.max(1, x(r.baselinePct) - LEFT)}
                height="12"
                fill="var(--chart-baseline)"
                opacity="0.85"
              />
              <text x={x(r.successPct) + 6} y={y + 14} fontSize="11" fill="var(--fg-hi)" fontFamily="var(--font-mono)">
                {r.successPct.toFixed(1)}
              </text>
            </g>
          )
        })}
      </svg>
      <ul className="legend">
        <li>
          <span className="swatch" style={{ background: 'var(--chart-flow)', borderColor: 'var(--chart-flow)' }} /> flow
        </li>
        <li>
          <span className="swatch" style={{ background: 'var(--chart-baseline)', borderColor: 'var(--chart-baseline)' }} /> baseline
        </li>
      </ul>
    </div>
  )
}
