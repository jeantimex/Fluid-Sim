import './style.css'
import { sim2dConfig } from './sim2dConfig.js'

const app = document.querySelector('#app')
const canvas = document.createElement('canvas')
const ctx = canvas.getContext('2d')
canvas.className = 'sim-canvas'
app.appendChild(canvas)

function resize() {
  const dpr = Math.max(1, window.devicePixelRatio || 1)
  const width = window.innerWidth
  const height = window.innerHeight
  canvas.width = Math.floor(width * dpr)
  canvas.height = Math.floor(height * dpr)
  canvas.style.width = `${width}px`
  canvas.style.height = `${height}px`
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
}

function frame() {
  ctx.fillStyle = sim2dConfig.render.clearColor
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  requestAnimationFrame(frame)
}

window.addEventListener('resize', resize)
resize()
frame()
