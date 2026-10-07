import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer,
} from 'recharts'
import { fetchStats } from '../api/api'

const METRIC_KEYS = ['accuracy', 'precision', 'recall', 'f1']

function StatCard({ label, value, sub }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 flex flex-col gap-1">
      <p className="text-xs text-gray-400 uppercase tracking-wide font-medium">{label}</p>
      <p className="text-2xl font-bold text-gray-900">{value}</p>
      {sub && <p className="text-xs text-gray-500">{sub}</p>}
    </div>
  )
}

function MetricsTable({ models, bestModel }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-gray-200">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-gray-50 border-b border-gray-200">
            <th className="px-4 py-3 text-left font-medium text-gray-600">Model</th>
            {METRIC_KEYS.map((k) => (
              <th key={k} className="px-4 py-3 text-right font-medium text-gray-600 capitalize">
                {k === 'f1' ? 'F1' : k.charAt(0).toUpperCase() + k.slice(1)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr
              key={m.name}
              className={`border-b border-gray-100 last:border-0 transition-colors ${
                m.name === bestModel ? 'bg-blue-50' : 'hover:bg-gray-50'
              }`}
            >
              <td className="px-4 py-3 font-medium text-gray-800 whitespace-nowrap">
                {m.name === bestModel && (
                  <span className="mr-1.5 text-blue-500 text-xs font-bold">★</span>
                )}
                {m.name}
              </td>
              {METRIC_KEYS.map((k) => (
                <td key={k} className="px-4 py-3 text-right tabular-nums text-gray-700">
                  {(m[k] * 100).toFixed(2)}%
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ConfusionMatrix({ matrix, modelName }) {
  if (!matrix || matrix.length < 2) return null
  const [[tn, fp], [fn, tp]] = matrix
  const total = tn + fp + fn + tp
  const cells = [
    { label: 'True Negative', value: tn, bg: 'bg-green-100', text: 'text-green-800' },
    { label: 'False Positive', value: fp, bg: 'bg-red-100', text: 'text-red-800' },
    { label: 'False Negative', value: fn, bg: 'bg-red-100', text: 'text-red-800' },
    { label: 'True Positive', value: tp, bg: 'bg-green-100', text: 'text-green-800' },
  ]
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5">
      <h3 className="font-semibold text-gray-800 mb-1">Confusion Matrix</h3>
      <p className="text-xs text-gray-400 mb-4">{modelName} · test set ({total.toLocaleString()} samples)</p>
      <div className="grid grid-cols-2 gap-2 max-w-xs">
        {cells.map((c) => (
          <div key={c.label} className={`rounded-lg p-3 ${c.bg}`}>
            <p className={`text-xs font-medium mb-1 ${c.text}`}>{c.label}</p>
            <p className={`text-xl font-bold ${c.text}`}>{c.value.toLocaleString()}</p>
            <p className={`text-xs ${c.text} opacity-70`}>
              {((c.value / total) * 100).toFixed(1)}%
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [stats, setStats]   = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]   = useState(null)

  useEffect(() => {
    fetchStats()
      .then(setStats)
      .catch((err) => setError(err.message || 'Failed to load stats'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32 gap-3 text-gray-500">
        <svg className="animate-spin w-5 h-5" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" />
          <path fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 100 16v-4l-3 3 3 3v-4a8 8 0 01-8-8z" className="opacity-75" />
        </svg>
        Loading stats…
      </div>
    )
  }

  if (error) {
    return (
      <div className="max-w-xl mx-auto px-4 py-16 text-center">
        <p className="text-red-600 font-medium mb-1">Failed to load dashboard data</p>
        <p className="text-sm text-gray-500">{error}</p>
        <p className="text-xs text-gray-400 mt-3">Make sure the FastAPI backend is running.</p>
      </div>
    )
  }

  const { dataset, models, best_model } = stats

  // Bar chart data: one entry per model, all metrics
  const barData = models.map((m) => ({
    name: m.name.replace('Multinomial ', 'MN ').replace('Logistic ', 'LR '),
    Accuracy:  +(m.accuracy  * 100).toFixed(2),
    Precision: +(m.precision * 100).toFixed(2),
    Recall:    +(m.recall    * 100).toFixed(2),
    F1:        +(m.f1        * 100).toFixed(2),
  }))

  // Radar chart data for best model
  const bestModelData = models.find((m) => m.name === best_model) || models[0]
  const radarData = METRIC_KEYS.map((k) => ({
    metric: k === 'f1' ? 'F1' : k.charAt(0).toUpperCase() + k.slice(1),
    value:  +(bestModelData[k] * 100).toFixed(2),
  }))

  return (
    <div className="max-w-5xl mx-auto px-4 py-10 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 mb-1">Model Dashboard</h1>
        <p className="text-sm text-gray-500">
          Evaluation results from the trained ML pipeline.
          Best model: <span className="font-semibold text-blue-600">{best_model}</span>
        </p>
      </div>

      {/* Dataset stat cards */}
      <section>
        <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">Dataset</h2>
        <div className="grid grid-cols-3 gap-4">
          <StatCard
            label="Total Articles"
            value={dataset.total.toLocaleString()}
            sub="After preprocessing"
          />
          <StatCard
            label="Fake Articles"
            value={dataset.fake_count.toLocaleString()}
            sub={`${((dataset.fake_count / dataset.total) * 100).toFixed(1)}% of dataset`}
          />
          <StatCard
            label="Real Articles"
            value={dataset.real_count.toLocaleString()}
            sub={`${((dataset.real_count / dataset.total) * 100).toFixed(1)}% of dataset`}
          />
        </div>
      </section>

      {/* Bar chart: all models vs all metrics */}
      <section>
        <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
          Model Comparison
        </h2>
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis domain={[90, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => `${v}%`} />
              <Legend />
              <Bar dataKey="Accuracy"  fill="#3b82f6" radius={[3,3,0,0]} />
              <Bar dataKey="Precision" fill="#10b981" radius={[3,3,0,0]} />
              <Bar dataKey="Recall"    fill="#f59e0b" radius={[3,3,0,0]} />
              <Bar dataKey="F1"        fill="#8b5cf6" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Bottom row: radar + confusion matrix */}
      <section>
        <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
          Best Model — {best_model}
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Radar */}
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <h3 className="font-semibold text-gray-800 mb-4">Metric Profile</h3>
            <ResponsiveContainer width="100%" height={220}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
                <PolarRadiusAxis angle={30} domain={[95, 100]} tick={{ fontSize: 10 }} />
                <Radar name={best_model} dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* Confusion matrix */}
          <ConfusionMatrix
            matrix={bestModelData.confusion_matrix}
            modelName={best_model}
          />
        </div>
      </section>

      {/* Full metrics table */}
      <section>
        <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
          All Models
        </h2>
        <MetricsTable models={models} bestModel={best_model} />
      </section>
    </div>
  )
}
