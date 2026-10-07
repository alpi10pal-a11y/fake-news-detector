import { useState } from 'react'
import { predict } from '../api/api'

const EXAMPLE_FAKE =
  'Scientists SHOCKED: Government hiding cure for cancer! Big Pharma exposed in massive conspiracy that will blow your mind. Share before they DELETE this!'

const EXAMPLE_REAL =
  'The Federal Reserve raised its benchmark interest rate by a quarter percentage point on Wednesday, marking the central bank\'s ongoing effort to bring inflation back toward its 2 percent target.'

function ResultBadge({ prediction, score, scoreType, modelUsed }) {
  const isFake = prediction === 'FAKE'
  const pct    = scoreType === 'confidence' ? `${(score * 100).toFixed(1)}%` : score.toFixed(3)

  return (
    <div
      className={`rounded-xl border-2 p-5 mt-6 ${
        isFake
          ? 'border-red-300 bg-red-50'
          : 'border-green-300 bg-green-50'
      }`}
    >
      <div className="flex items-center gap-3 mb-3">
        <span className="text-3xl">{isFake ? '🚨' : '✅'}</span>
        <span
          className={`text-2xl font-bold tracking-wide ${
            isFake ? 'text-red-700' : 'text-green-700'
          }`}
        >
          {prediction}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="bg-white rounded-lg p-3 border border-gray-200">
          <p className="text-gray-500 mb-0.5">
            {scoreType === 'confidence' ? 'Confidence' : 'Decision Score'}
          </p>
          <p className={`font-semibold text-lg ${isFake ? 'text-red-600' : 'text-green-600'}`}>
            {pct}
          </p>
        </div>
        <div className="bg-white rounded-lg p-3 border border-gray-200">
          <p className="text-gray-500 mb-0.5">Model</p>
          <p className="font-semibold text-gray-800">{modelUsed}</p>
        </div>
      </div>

      {/* Confidence bar */}
      {scoreType === 'confidence' && (
        <div className="mt-3">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>REAL</span>
            <span>FAKE</span>
          </div>
          <div className="h-2 rounded-full bg-gray-200 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                isFake ? 'bg-red-500' : 'bg-green-500'
              }`}
              style={{ width: `${score * 100}%` }}
            />
          </div>
        </div>
      )}
    </div>
  )
}

export default function Detector() {
  const [text, setText]     = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    if (!text.trim()) return
    setLoading(true)
    setResult(null)
    setError(null)
    try {
      const data = await predict(text)
      setResult(data)
    } catch (err) {
      const detail = err.response?.data?.detail
      setError(
        typeof detail === 'string'
          ? detail
          : 'Could not reach the API. Make sure the backend is running.'
      )
    } finally {
      setLoading(false)
    }
  }

  function loadExample(exampleText) {
    setText(exampleText)
    setResult(null)
    setError(null)
  }

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0

  return (
    <div className="max-w-2xl mx-auto px-4 py-10">
      <div className="mb-6 text-center">
        <h1 className="text-2xl font-bold text-gray-900 mb-1">
          News Article Classifier
        </h1>
        <p className="text-gray-500 text-sm">
          Paste a headline or full article. The model will classify it as{' '}
          <span className="font-medium text-red-600">FAKE</span> or{' '}
          <span className="font-medium text-green-600">REAL</span>.
        </p>
      </div>

      {/* Example buttons */}
      <div className="flex gap-2 mb-3 flex-wrap">
        <span className="text-xs text-gray-400 self-center">Try an example:</span>
        <button
          onClick={() => loadExample(EXAMPLE_FAKE)}
          className="text-xs px-2.5 py-1 rounded-full border border-red-200 text-red-600 hover:bg-red-50 transition-colors cursor-pointer"
        >
          Fake article
        </button>
        <button
          onClick={() => loadExample(EXAMPLE_REAL)}
          className="text-xs px-2.5 py-1 rounded-full border border-green-200 text-green-600 hover:bg-green-50 transition-colors cursor-pointer"
        >
          Real article
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="relative">
          <textarea
            value={text}
            onChange={(e) => { setText(e.target.value); setResult(null); setError(null) }}
            rows={8}
            placeholder="Paste your news article or headline here…"
            className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none transition-shadow"
          />
          {wordCount > 0 && (
            <span className="absolute bottom-2 right-3 text-xs text-gray-400">
              {wordCount} word{wordCount !== 1 ? 's' : ''}
            </span>
          )}
        </div>

        <div className="flex gap-2">
          <button
            type="submit"
            disabled={loading || !text.trim()}
            className="flex-1 py-2.5 rounded-xl bg-blue-600 text-white text-sm font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" />
                  <path fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 100 16v-4l-3 3 3 3v-4a8 8 0 01-8-8z" className="opacity-75" />
                </svg>
                Analysing…
              </span>
            ) : 'Analyse Article'}
          </button>
          {text && (
            <button
              type="button"
              onClick={() => { setText(''); setResult(null); setError(null) }}
              className="px-4 py-2.5 rounded-xl border border-gray-300 text-sm text-gray-600 hover:bg-gray-100 transition-colors cursor-pointer"
            >
              Clear
            </button>
          )}
        </div>
      </form>

      {/* Error */}
      {error && (
        <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          <span className="font-medium">Error: </span>{error}
        </div>
      )}

      {/* Result */}
      {result && (
        <ResultBadge
          prediction={result.prediction}
          score={result.score}
          scoreType={result.score_type}
          modelUsed={result.model_used}
        />
      )}
    </div>
  )
}
