import { useEffect, useMemo, useRef, useState } from 'react'
import AgentSection from './components/AgentSection'
import AgreementSection from './components/AgreementSection'
import EvidenceSection from './components/EvidenceSection'
import InputSection from './components/InputSection'
import RiskSection from './components/RiskSection'
import ThinkingPipeline from './components/ThinkingPipeline'
import VerdictCard from './components/VerdictCard'
import SoftAuroraBackground from './components/background/SoftAuroraBackground'
import { normalizeAnalysisResult } from './lib/normalizeAnalysis'
import './index.css'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
const MIN_WORDS = 50
const RESULT_REVEAL_ORDER = ['agreement', 'agents', 'evidence', 'risks', 'debug']
const PIPELINE_INTERVAL_MS = 900
const FINAL_REVEAL_DELAY_MS = 520

const MODE_CONFIG = {
  agentic: {
    label: 'Agentic Analysis',
    badge: 'Multi-Agent Fact Review',
    title: "Let's see if this story actually checks out",
    subtitle:
      'Three reasoning agents dig through the evidence, cross-check each other, and deliver a transparent verdict you can trust.',
    thinkingTitle: 'Three reasoning agents are on the case',
    thinkingBody:
      'Each stage lights up as the pipeline scores the article, pulls retrieval context, and synthesizes a final call.',
    steps: [
      { label: 'Running ML model', model: 'Logistic Regression + TF-IDF', provider: 'scikit-learn' },
      { label: 'Retrieving relevant documents', model: 'ChromaDB semantic search', provider: 'Vector Store' },
      { label: 'Agent A — Conservative reasoning', model: 'llama-3.3-70b-versatile', provider: 'Groq' },
      { label: 'Agent B — Skeptical reasoning', model: 'llama-3.3-70b-versatile', provider: 'Groq' },
      { label: 'Agent C — Neutral reasoning', model: 'llama-3.3-70b-versatile', provider: 'Groq' },
      { label: 'Synthesizing final verdict', model: 'meta/llama-3.3-70b-instruct', provider: 'NVIDIA' },
    ],
  },
  ml: {
    label: 'Classical ML',
    badge: 'ML-First Review',
    title: 'Quick model check — straight to the verdict',
    subtitle:
      "Same pipeline, but we lead with the classifier's signal and let the agents weigh in as backup.",
    thinkingTitle: 'The classifier is crunching the numbers',
    thinkingBody:
      'We start with the model signal, then cross-check retrieval and the agent findings.',
    steps: [
      { label: 'Preparing article features', model: 'TF-IDF vectorizer', provider: 'scikit-learn' },
      { label: 'Running logistic model', model: 'Logistic Regression', provider: 'scikit-learn' },
      { label: 'Calibrating confidence', model: 'Probability calibration', provider: 'scikit-learn' },
      { label: 'Preparing final verdict', model: 'Classifier output', provider: 'scikit-learn' },
    ],
  },
}

function App() {
  const [mode, setMode] = useState('agentic')
  const [input, setInput] = useState('')
  const [status, setStatus] = useState('idle')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [pipelineStepIndex, setPipelineStepIndex] = useState(0)
  const [revealedSections, setRevealedSections] = useState([])
  const [pendingResult, setPendingResult] = useState(null)
  const [countdown, setCountdown] = useState(5)

  const resultsRef = useRef(null)

  const currentMode = MODE_CONFIG[mode]
  const pipelineSteps = currentMode.steps
  const wordCount = useMemo(() => {
    const trimmed = input.trim()
    return trimmed ? trimmed.split(/\s+/).length : 0
  }, [input])
  const isInputTooShort = wordCount > 0 && wordCount < MIN_WORDS
  const isLoading = status === 'loading'
  const isReportReady = status === 'ready' && result
  const isResultReady = status === 'result' && result

  useEffect(() => {
    if (!isLoading) {
      setPipelineStepIndex(0)
      return
    }

    setPipelineStepIndex(0)
    const stepCount = pipelineSteps.length
    const timeouts = pipelineSteps.map((_, index) =>
      window.setTimeout(() => {
        setPipelineStepIndex(index)
      }, index * PIPELINE_INTERVAL_MS),
    )

    const completionTimeout = window.setTimeout(() => {
      setPipelineStepIndex(stepCount)
    }, stepCount * PIPELINE_INTERVAL_MS)

    return () => {
      timeouts.forEach((timeout) => window.clearTimeout(timeout))
      window.clearTimeout(completionTimeout)
    }
  }, [isLoading, pipelineSteps])

  useEffect(() => {
    if (!isLoading || !pendingResult) {
      return
    }

    if (pipelineStepIndex < pipelineSteps.length) {
      return
    }

    const revealTimeout = window.setTimeout(() => {
      setResult(pendingResult)
      setPendingResult(null)
      setStatus('ready')
      setCountdown(5)
    }, FINAL_REVEAL_DELAY_MS)

    return () => {
      window.clearTimeout(revealTimeout)
    }
  }, [isLoading, pendingResult, pipelineStepIndex, pipelineSteps.length])

  useEffect(() => {
    if (!isReportReady) {
      return
    }

    if (countdown <= 0) {
      setStatus('result')
      return
    }

    const timer = window.setTimeout(() => {
      setCountdown((current) => current - 1)
    }, 1000)

    return () => {
      window.clearTimeout(timer)
    }
  }, [isReportReady, countdown])

  useEffect(() => {
    if (!isResultReady) {
      setRevealedSections([])
      return
    }

    const timers = RESULT_REVEAL_ORDER.map((section, index) =>
      window.setTimeout(() => {
        setRevealedSections((current) => [...current, section])
      }, 240 * (index + 1)),
    )

    return () => {
      timers.forEach((timer) => window.clearTimeout(timer))
    }
  }, [isResultReady])

  useEffect(() => {
    if (!isResultReady || !resultsRef.current) {
      return
    }

    resultsRef.current.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    })
  }, [isResultReady])

  function handleModeChange(nextMode) {
    if (isLoading || nextMode === mode) {
      return
    }

    setMode(nextMode)
    setStatus('idle')
    setResult(null)
    setPendingResult(null)
    setError('')
    setRevealedSections([])
    setPipelineStepIndex(0)
  }

  function sectionIsVisible(name) {
    return revealedSections.includes(name)
  }

  function handleReset() {
    setStatus('idle')
    setResult(null)
    setPendingResult(null)
    setError('')
    setRevealedSections([])
    setPipelineStepIndex(0)
    setInput('')
    setCountdown(5)
    if (typeof window !== 'undefined') {
      window.scrollTo({ top: 0, behavior: 'smooth' })
    }
  }

  function handleShowReport() {
    setStatus('result')
    setCountdown(0)
  }

  async function handleSubmit(event) {
    event.preventDefault()

    if (!input.trim()) {
      setError('Paste a news article or claim before starting the analysis.')
      return
    }

    if (wordCount < MIN_WORDS) {
      setError(`Please provide at least ${MIN_WORDS} words so the system has enough context to analyze.`)
      return
    }

    setStatus('loading')
    setError('')
    setResult(null)
    setPendingResult(null)
    setRevealedSections([])
    setPipelineStepIndex(0)
    setCountdown(5)

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: input, mode }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data?.detail || data?.error || 'The analysis request failed. Please try again.')
      }

      setPendingResult(normalizeAnalysisResult(data, mode))
    } catch (requestError) {
      setError(requestError.message || 'Unable to analyze this article right now.')
      setPendingResult(null)
      setStatus('idle')
    }
  }

  return (
    <div className="relative min-h-screen overflow-hidden text-[#ececf1]">
      <SoftAuroraBackground
        colorStops={['#7FB3D5', '#9FD8C9', '#E0A8C8']}
        amplitude={1.1}
        blend={0.55}
        speed={0.8}
      />

      <main className="relative z-10 mx-auto flex min-h-screen w-full max-w-6xl flex-col px-4 py-6 sm:px-6 sm:py-8 lg:px-8 lg:py-10">
        {!isLoading && !isReportReady && !isResultReady ? (
          <section
            className={`mx-auto flex w-full max-w-4xl flex-col items-center text-center ${
              status === 'idle' ? 'flex-1 justify-center pt-8 sm:pt-12' : 'pt-8 sm:pt-10'
            }`}
          >
            <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-[#262629] bg-[#151517]/80 py-1 pl-1 pr-4 text-[12px] text-[#a1a1aa] backdrop-blur-xl">
              <span className="rounded-full bg-[#ececf1] px-2.5 py-0.5 text-[11px] font-semibold text-[#0d0d0f]">
                AI
              </span>
              <span>Multi-agent news verification</span>
            </div>
            <h1 className="max-w-4xl text-5xl font-bold tracking-tight text-[#ececf1] sm:text-6xl lg:text-7xl">
              News Credibility Analyst
            </h1>
            <p className="mt-6 max-w-2xl text-[15px] leading-7 text-[#a1a1aa] sm:text-base">
              Paste any news article and let three reasoning agents cross-check the evidence, weigh the ML signal, and deliver a transparent verdict you can trust.
            </p>

            <div className="mt-10 w-full">
              <InputSection
                value={input}
                disabled={isLoading}
                onChange={setInput}
                onSubmit={handleSubmit}
                minWords={MIN_WORDS}
                wordCount={wordCount}
                isInputTooShort={isInputTooShort}
                mode={mode}
                onModeChange={handleModeChange}
              />
            </div>
          </section>
        ) : null}

        {isLoading ? (
          <section className="mx-auto flex min-h-[calc(100vh-4rem)] w-full max-w-3xl flex-col justify-center">
            <ThinkingPipeline
              title={currentMode.thinkingTitle}
              description={currentMode.thinkingBody}
              steps={pipelineSteps}
              activeStepIndex={pipelineStepIndex}
            />
          </section>
        ) : null}

        {isReportReady ? (
          <section className="mx-auto flex min-h-[calc(100vh-4rem)] w-full max-w-2xl flex-col items-center justify-center text-center">
            <div className="relative w-full">
              <div
                aria-hidden="true"
                className="pointer-events-none absolute -inset-2 rounded-[32px] bg-gradient-to-r from-[#7FB3D5]/25 via-[#9FD8C9]/30 to-[#E0A8C8]/25 opacity-70 blur-2xl"
              />
              <div className="relative overflow-hidden rounded-[28px] border border-[#3a3a42] bg-gradient-to-b from-[#2a2a30]/95 to-[#1f1f23]/95 p-8 shadow-[0_30px_80px_-20px_rgba(0,0,0,0.6)] backdrop-blur-2xl sm:p-10">
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-x-10 top-0 h-px bg-gradient-to-r from-transparent via-[#9FD8C9]/60 to-transparent"
                />

                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl border border-[#6BAAA5]/50 bg-[#1a2a2f]">
                  <span className="absolute flex h-16 w-16 animate-ping rounded-2xl bg-[#9FD8C9]/10" />
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="relative h-7 w-7 text-[#9FD8C9]"
                    aria-hidden="true"
                  >
                    <path d="m5 12 5 5L20 7" />
                  </svg>
                </div>

                <p className="mt-5 text-[11px] font-semibold tracking-[0.24em] text-[#9FD8C9] uppercase">
                  Analysis complete
                </p>
                <h2 className="mt-2 text-2xl font-bold text-[#ececf1] sm:text-3xl">
                  Your credibility report is ready
                </h2>
                <p className="mx-auto mt-3 max-w-md text-[14px] leading-6 text-[#a1a1aa]">
                  All agents have delivered their verdicts. Opening your full report in{' '}
                  <span className="font-semibold text-[#ececf1] tabular-nums">{countdown}s</span>…
                </p>

                <div className="mx-auto mt-5 h-1 w-full max-w-xs overflow-hidden rounded-full bg-[#1d1d21]">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-[#7FB3D5] via-[#9FD8C9] to-[#E0A8C8] transition-all duration-1000 ease-linear"
                    style={{ width: `${((5 - countdown) / 5) * 100}%` }}
                  />
                </div>

                <div className="mt-7 flex flex-col items-center justify-center gap-3 sm:flex-row">
                  <button
                    type="button"
                    onClick={handleShowReport}
                    className="inline-flex items-center gap-2 rounded-lg bg-gradient-to-b from-[#ffffff] to-[#d4d4d8] px-5 py-2 text-[14px] font-semibold text-[#0d0d0f] shadow-[0_4px_14px_-2px_rgba(255,255,255,0.25)] transition-all duration-200 hover:shadow-[0_6px_18px_-2px_rgba(255,255,255,0.35)] hover:brightness-105 active:brightness-95"
                  >
                    <span>View Report</span>
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="h-3.5 w-3.5"
                      aria-hidden="true"
                    >
                      <path d="M5 12h14" />
                      <path d="m12 5 7 7-7 7" />
                    </svg>
                  </button>
                  <button
                    type="button"
                    onClick={handleReset}
                    className="inline-flex items-center gap-2 rounded-lg border border-[#3a3a42] bg-[#2a2a30]/80 px-5 py-2 text-[14px] font-semibold text-[#ececf1] backdrop-blur-xl transition-all duration-200 hover:border-[#5a7a95] hover:bg-[#33333a]/80"
                  >
                    Start over
                  </button>
                </div>
              </div>
            </div>
          </section>
        ) : null}

        {error ? (
          <section className="mx-auto mt-6 w-full max-w-3xl rounded-[28px] border border-[#3a2525] bg-[#241818] p-5 text-left shadow-[0_20px_60px_-20px_rgba(201,115,112,0.35)] backdrop-blur-xl">
            <div className="flex items-center gap-2">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-3.5 w-3.5 text-[#c97370]"
                aria-hidden="true"
              >
                <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z" />
                <path d="M12 9v4" />
                <path d="M12 17h.01" />
              </svg>
              <p className="text-[11px] font-semibold tracking-[0.24em] text-[#c97370] uppercase">
                Something went wrong
              </p>
            </div>
            <p className="mt-2 text-[13px] leading-7 text-[#e0b0ae]">{error}</p>
          </section>
        ) : null}

        {isResultReady ? (
          <section ref={resultsRef} className="mx-auto flex w-full max-w-6xl flex-col gap-6 pt-4 sm:gap-8">
            <div className="flex items-center justify-between gap-4">
              <div className="inline-flex items-center gap-2 rounded-full border border-[#262629] bg-[#151517]/80 py-1 pl-1 pr-4 text-[12px] text-[#a1a1aa] backdrop-blur-xl">
                <span className="rounded-full bg-[#9FD8C9] px-2.5 py-0.5 text-[11px] font-semibold text-[#0d0d0f]">
                  ✓
                </span>
                <span>Analysis complete</span>
              </div>
              <button
                type="button"
                onClick={handleReset}
                className="inline-flex items-center gap-1.5 rounded-lg border border-[#3a3a42] bg-[#2a2a30]/80 px-4 py-1.5 text-[13px] font-semibold text-[#ececf1] backdrop-blur-xl transition-all duration-200 hover:border-[#5a7a95] hover:bg-[#33333a]/80"
              >
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-3.5 w-3.5"
                  aria-hidden="true"
                >
                  <path d="M12 5v14" />
                  <path d="M5 12h14" />
                </svg>
                New Analysis
              </button>
            </div>
            <VerdictCard final={result.final} agreement={result.agreement} mode={mode} mlSignal={result.debug.mlSignal} />

            {mode === 'agentic' ? (
              <>
                <div
                  className={`transform transition-all duration-700 ease-out ${
                    sectionIsVisible('agreement')
                      ? 'translate-y-0 opacity-100'
                      : 'pointer-events-none translate-y-4 opacity-0'
                  }`}
                >
                  <AgreementSection agreement={result.agreement} mode={mode} mlSignal={result.debug.mlSignal} />
                </div>

                <div
                  className={`transform transition-all duration-700 ease-out ${
                    sectionIsVisible('agents')
                      ? 'translate-y-0 opacity-100'
                      : 'pointer-events-none translate-y-4 opacity-0'
                  }`}
                >
                  <AgentSection agents={result.agents} mode={mode} />
                </div>

                <div
                  className={`transform transition-all duration-700 ease-out ${
                    sectionIsVisible('evidence')
                      ? 'translate-y-0 opacity-100'
                      : 'pointer-events-none translate-y-4 opacity-0'
                  }`}
                >
                  <EvidenceSection evidence={result.evidence} mode={mode} />
                </div>
              </>
            ) : null}

            <div
              className={`transform transition-all duration-700 ease-out ${
                sectionIsVisible('risks')
                  ? 'translate-y-0 opacity-100'
                  : 'pointer-events-none translate-y-4 opacity-0'
              }`}
            >
              <RiskSection riskFactors={result.riskFactors} />
            </div>

            {mode === 'agentic' ? (
              <details
                className={`group rounded-[28px] border border-[#3a3a42] bg-gradient-to-b from-[#2a2a30]/95 to-[#1f1f23]/95 p-5 shadow-[0_20px_60px_-20px_rgba(0,0,0,0.55)] backdrop-blur-xl transition-all duration-700 ease-out ${
                  sectionIsVisible('debug')
                    ? 'translate-y-0 opacity-100'
                    : 'pointer-events-none translate-y-4 opacity-0'
                }`}
              >
                <summary className="flex cursor-pointer list-none items-center gap-2 text-[13px] font-semibold text-[#a1a1aa] transition group-open:text-[#ececf1]">
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="h-3.5 w-3.5 transition-transform duration-200 group-open:rotate-90"
                    aria-hidden="true"
                  >
                    <path d="m9 18 6-6-6-6" />
                  </svg>
                  <span>Peek under the hood</span>
                </summary>
                <div className="mt-5 grid gap-4 sm:grid-cols-2">
                  <div className="rounded-2xl border border-[#262629] bg-[#1c1c1f] p-4">
                    <p className="text-[11px] font-semibold tracking-[0.24em] text-[#9FD8C9] uppercase">
                      Dominant Agent
                    </p>
                    <p className="mt-2 text-base font-semibold text-[#ececf1]">
                      {result.final.dominantAgent}
                    </p>
                  </div>
                  <div className="rounded-2xl border border-[#262629] bg-[#1c1c1f] p-4">
                    <p className="text-[11px] font-semibold tracking-[0.24em] text-[#a8b8e0] uppercase">
                      Conflict Resolution
                    </p>
                    <p className="mt-2 text-[13px] leading-7 text-[#a1a1aa]">
                      {result.final.conflict}
                    </p>
                  </div>
                </div>
              </details>
            ) : null}
          </section>
        ) : null}
      </main>
    </div>
  )
}

export default App
