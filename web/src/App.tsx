import { AblationBars } from './components/story/AblationBars'
import { ConditionScatter } from './components/story/ConditionScatter'
import { ResultBento } from './components/story/ResultBento'
import { StackGrid } from './components/story/StackGrid'
import { StoryBeat } from './components/story/StoryBeat'
import {
  ABLATION,
  DECISIONS,
  ILLUSTRATIVE,
  METRICS,
  NEXT,
  PROTOCOL,
  SECTORS,
} from './data'

const TOC = [
  { href: '#problem', label: 'The problem' },
  { href: '#answer', label: 'The answer' },
  { href: '#result', label: 'The result' },
  { href: '#stack', label: 'Tech stack' },
  { href: '#decisions', label: 'Design decisions' },
  { href: '#next', label: 'Next and real world' },
  { href: '#close', label: 'Close' },
]

const NOTE = ILLUSTRATIVE
  ? ' These numbers are illustrative until a measured run overwrites data.ts.'
  : ''

export function App() {
  return (
    <>
      <a className="skip-link" href="#problem">
        Skip to the walkthrough
      </a>

      {ILLUSTRATIVE ? (
        <div className="banner" role="status">
          Illustrative numbers. Do not read these as a measured flow fit.
        </div>
      ) : null}

      <div className="masthead">
        <div className="masthead__inner">
          <div className="masthead__mark">
            <b>FlowMatch</b> · new draws under conditions the model did not train on
          </div>
          <ul className="masthead__nav">
            <li>
              <a href="#problem">problem</a>
            </li>
            <li>
              <a href="#result">result</a>
            </li>
            <li>
              <a href="#decisions">decisions</a>
            </li>
            <li>
              <a href="#next">next</a>
            </li>
          </ul>
        </div>
      </div>

      <main className="page">
        <header className="page-hero">
          <p className="meta">Walkthrough · teach a sampler a toy cloud, then ask for new conditions</p>
          <h1>FlowMatch</h1>
          <p className="lead">
            I teach a small generative model to draw points from a known toy cloud given a condition
            code. Then I give it conditions it did not train on and ask whether the new points land
            where the true cloud says they should. The number I trust is that hit rate, sitting next
            to a naive random cloud in the same plane.
          </p>
          <p className="intro-detail">
            Headline metric is how often a new draw lands inside a ball of radius {PROTOCOL.tau}{' '}
            around the true mean, on conditions I held out. Floor is {PROTOCOL.floorPct}%. I print the
            sampler next to a naive random cloud that ignores the condition, same sample budget, same split.
            {ILLUSTRATIVE
              ? ' Numbers below are labeled illustrative.'
              : ` Measured hit rate is ${METRICS.successPct}%. The naive cloud hits ${METRICS.baselinePct}%.`}
          </p>
          <nav aria-label="On this page">
            <ul className="toc">
              {TOC.map((item) => (
                <li key={item.href}>
                  <a href={item.href}>{item.label}</a>
                </li>
              ))}
            </ul>
          </nav>
        </header>

        <StoryBeat
          id="problem"
          kicker="The problem"
          title="A blob that ignores the condition will miss the target"
          caption={`Eight clusters on a circle. Train conditions ${PROTOCOL.trainConds.join(', ')}; holdout ${PROTOCOL.holdConds.join(', ')}. The dashed rings are the balls I score.${NOTE}`}
          visual={<ConditionScatter />}
        >
          <p>
            The data is a 2D mixture I fully control. Condition c picks an angle, and x is a tight
            cluster around a point on a circle of radius {PROTOCOL.radius}. If I sample without
            looking at c, most draws miss the ball that belongs to that condition.
          </p>
          <p>
            Likelihood on train points is a weak check here. A model can look good on the points it
            trained on and still fail when I ask it to sample under a condition I held out. The
            number I care about is the hit rate on those held-out conditions.
          </p>
        </StoryBeat>

        <StoryBeat
          id="answer"
          kicker="The answer"
          title="Learn a path from noise to data, given the condition"
          caption={`Start at noise, take ${PROTOCOL.steps} steps along a velocity net of width ${PROTOCOL.width}. Condition is encoded as (cos θ, sin θ).${NOTE}`}
          visual={
            <div className="teach-card">
              <h3 className="teach-card__title">What gets trained, what gets scored</h3>
              <dl className="stat-grid">
                <div>
                  <dt>Train points</dt>
                  <dd>{PROTOCOL.nTrain}</dd>
                </div>
                <div>
                  <dt>Holdout samples</dt>
                  <dd>{PROTOCOL.nSample}</dd>
                </div>
                <div>
                  <dt>Train conditions</dt>
                  <dd>{PROTOCOL.trainConds.join(', ')}</dd>
                </div>
                <div>
                  <dt>Holdout conditions</dt>
                  <dd>{PROTOCOL.holdConds.join(', ')}</dd>
                </div>
              </dl>
              <p className="meta" style={{ textTransform: 'none', letterSpacing: 0 }}>
                Seed {PROTOCOL.seed}. {PROTOCOL.epochs} epochs. If a holdout condition appears in
                train, the leak test fails.
              </p>
            </div>
          }
        >
          <p>
            I train a thin velocity network the rectified-flow way: pick a noise point and a data
            point, walk the straight line between them, and ask the net to predict that velocity,
            given the condition. Sampling is the same walk run forward from noise, with the condition
            held fixed.
          </p>
          <p>
            Two conditions stay out of train. At test time I sample under those, and I count a hit
            when the point falls inside the true ball. The baseline is one round cloud fit to
            all train x, no condition, same number of draws.
          </p>
        </StoryBeat>

        <StoryBeat
          id="result"
          kicker="The result"
          title="New draws land where the true cloud says. A cloud that ignores the condition does not."
          caption={`Sampler hit rate ${METRICS.successPct}% vs naive cloud ${METRICS.baselinePct}% on ${PROTOCOL.nSample} holdout samples. Ablation bars use the same split.${NOTE}`}
          visual={<AblationBars />}
        >
          <p>
            Here is the frozen protocol. The number is a percentage: how often a new draw lands in
            the right ball. I did not pick it from train likelihood.
          </p>
          <ResultBento />
          <p style={{ marginTop: 'var(--space-5)' }}>
            The scatter in the problem figure is the same draw: sampler points sit in the dashed holdout rings, the
            naive cloud sprays across the plane. Ablation bars keep that split and change one knob at
            a time. A 1-D angle encoding is the weakest of the sampler variants; it still clears the
            floor. Extra steps past {PROTOCOL.steps} do not buy much.
          </p>
          <table className="choice-table">
            <caption className="sr-only">Ablation hit rates on the same holdout split</caption>
            <thead>
              <tr>
                <th scope="col">Knob</th>
                <th scope="col">Flow %</th>
                <th scope="col">Baseline %</th>
              </tr>
            </thead>
            <tbody>
              {ABLATION.map((r) => (
                <tr key={r.tag}>
                  <td>{r.tag}</td>
                  <td>{r.successPct.toFixed(1)}</td>
                  <td>{r.baselinePct.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </StoryBeat>

        <section className="story-beat" id="stack">
          <p className="story-kicker">Tech stack</p>
          <h2>The tools, in plain terms</h2>
          <p className="stack-intro">
            Standard tools, so the numbers rerun from this repo. Each card is what it does, then how.
          </p>
          <StackGrid />
        </section>

        <StoryBeat
          id="decisions"
          kicker="Design decisions"
          title="The calls I made"
          caption="What I first reached for, and what I built instead."
          visual={
            <div className="teach-card">
              <h3 className="teach-card__title">First idea, and what I built</h3>
              <table className="choice-table">
                <caption className="sr-only">Design choices</caption>
                <thead>
                  <tr>
                    <th scope="col">First idea</th>
                    <th scope="col">What I built</th>
                  </tr>
                </thead>
                <tbody>
                  {DECISIONS.map((d) => (
                    <tr key={d.first}>
                      <td>{d.first}</td>
                      <td>{d.built}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          }
        >
          <p>
            I scored region hits on holdout conditions, not train NLL. Train likelihood is debug. The
            page reports the hit rate next to a sampler that never sees the condition.
          </p>
          <p>
            I kept the net small: 2D, width {PROTOCOL.width}, {PROTOCOL.epochs} epochs, laptop budget.
            A latent-diffusion stack would hide the question I am asking.
          </p>
        </StoryBeat>

        <section className="story-beat" id="next">
          <p className="story-kicker">Next, run, sectors</p>
          <h2>Where I would take it</h2>
          <ul className="stack-list" style={{ maxWidth: 'var(--measure)' }}>
            {NEXT.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
          <h3 style={{ marginTop: 'var(--space-6)' }}>Run it locally</h3>
          <ol className="stack-list" style={{ maxWidth: 'var(--measure)' }}>
            <li>
              From the repo root, run <code>python scripts/run.py</code>. It prints sampler vs baseline
              hit rate, writes <code>metrics.json</code>, and overwrites <code>web/src/data.ts</code>.
            </li>
            <li>
              Run <code>npm --prefix web run dev</code> to read this page. Numbers come from{' '}
              <code>data.ts</code>, not from a live API.
            </li>
          </ol>
          <h3 style={{ marginTop: 'var(--space-6)' }}>Named sectors</h3>
          <ul className="stack-list" style={{ maxWidth: 'var(--measure)' }}>
            {SECTORS.map((s) => (
              <li key={s.name}>
                <strong>{s.name}.</strong> {s.why}
              </li>
            ))}
          </ul>
        </section>

        <footer
          id="close"
          style={{
            borderTop: '1px solid var(--line-rule)',
            paddingTop: 'var(--space-6)',
            marginTop: 'var(--space-6)',
            color: 'var(--fg-low)',
            fontSize: 'var(--fs-sm)',
          }}
        >
          <p style={{ maxWidth: 'var(--measure)' }}>
            Synthetic 2D mixture, frozen holdout conditions {PROTOCOL.holdConds.join(' and ')}, seed{' '}
            {PROTOCOL.seed}. This is a portfolio check that a thin conditional sampler beats a naive
            cloud on a known density. It is not a production generator and not a likelihood
            leaderboard.
            {ILLUSTRATIVE ? ' The numbers here are placeholders until the pipeline replaces them.' : ''}
          </p>
        </footer>
      </main>
    </>
  )
}
