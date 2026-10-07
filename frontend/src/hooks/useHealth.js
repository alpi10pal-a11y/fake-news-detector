import { useEffect, useState } from 'react'
import { checkHealth } from '../api/api'

export function useHealth() {
  const [status, setStatus] = useState('checking') // 'ok' | 'error' | 'checking'

  useEffect(() => {
    checkHealth()
      .then(() => setStatus('ok'))
      .catch(() => setStatus('error'))
  }, [])

  return status
}
