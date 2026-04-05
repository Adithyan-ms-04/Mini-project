import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Load env file from current directory
  const env = loadEnv(mode, process.cwd(), '');
  const backendPort = env.VITE_BACKEND_PORT || '8000';

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/predict': `http://localhost:${backendPort}`
      }
    }
  }
})
