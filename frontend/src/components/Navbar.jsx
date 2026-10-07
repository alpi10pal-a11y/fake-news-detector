import { NavLink } from 'react-router-dom'
import { useHealth } from '../hooks/useHealth'

const statusDot = {
  ok:       'bg-green-500',
  error:    'bg-red-500',
  checking: 'bg-yellow-400 animate-pulse',
}
const statusLabel = {
  ok:       'API online',
  error:    'API offline',
  checking: 'Connecting…',
}

export default function Navbar() {
  const health = useHealth()

  return (
    <nav className="sticky top-0 z-50 bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between h-14">
        {/* Brand */}
        <div className="flex items-center gap-2">
          <span className="text-xl">🔍</span>
          <span className="font-semibold text-gray-900 text-sm sm:text-base">
            Fake News Detector
          </span>
        </div>

        {/* Nav links */}
        <div className="flex items-center gap-1">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-blue-50 text-blue-700'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              }`
            }
          >
            Detector
          </NavLink>
          <NavLink
            to="/dashboard"
            className={({ isActive }) =>
              `px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-blue-50 text-blue-700'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              }`
            }
          >
            Dashboard
          </NavLink>
        </div>

        {/* Health indicator */}
        <div className="flex items-center gap-1.5 text-xs text-gray-500">
          <span className={`w-2 h-2 rounded-full ${statusDot[health]}`} />
          <span className="hidden sm:inline">{statusLabel[health]}</span>
        </div>
      </div>
    </nav>
  )
}
