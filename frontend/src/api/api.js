import client from './client'

export const checkHealth = () =>
  client.get('/health').then((r) => r.data)

export const predict = (text) =>
  client.post('/predict', { text }).then((r) => r.data)

export const fetchStats = () =>
  client.get('/stats').then((r) => r.data)
