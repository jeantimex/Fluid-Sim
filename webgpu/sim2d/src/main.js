import './style.css'
import { sim2dConfig } from './sim2dConfig.js'

const app = document.querySelector('#app')
const gpuCanvas = document.createElement('canvas')
gpuCanvas.className = 'gpu-canvas'
app.appendChild(gpuCanvas)

const canvas = document.createElement('canvas')
const ctx = canvas.getContext('2d')
canvas.className = 'sim-canvas'
app.appendChild(canvas)

const state = {
  positions: [],
  velocities: [],
  viewSize: { width: 0, height: 0 },
  gpuStatus: 'init',
  gpuDevice: null,
  gpuContext: null,
  gpuFormat: null,
  gpuClearColor: { r: 0, g: 0, b: 0, a: 1 },
}

function parseHexColor(hex) {
  const value = hex.trim()
  const match = /^#([0-9a-fA-F]{6})$/.exec(value)
  if (!match) return { r: 0, g: 0, b: 0, a: 1 }
  const intValue = Number.parseInt(match[1], 16)
  return {
    r: ((intValue >> 16) & 255) / 255,
    g: ((intValue >> 8) & 255) / 255,
    b: (intValue & 255) / 255,
    a: 1,
  }
}

state.gpuClearColor = parseHexColor(sim2dConfig.render.clearColor)

function createRng(seed) {
  let t = seed >>> 0
  return () => {
    t += 0x6d2b79f5
    let r = t
    r = Math.imul(r ^ (r >>> 15), r | 1)
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61)
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296
  }
}

function calculateSpawnCountPerAxisBox2D(size, spawnDensity) {
  const area = size[0] * size[1]
  const targetTotal = Math.ceil(area * spawnDensity)
  const lenSum = size[0] + size[1]
  const t = [size[0] / lenSum, size[1] / lenSum]
  const m = Math.sqrt(targetTotal / (t[0] * t[1]))
  const nx = Math.ceil(t[0] * m)
  const ny = Math.ceil(t[1] * m)
  return [nx, ny]
}

function spawnInRegion(region, spawnDensity) {
  const centre = region.position
  const size = region.size
  const [nx, ny] = calculateSpawnCountPerAxisBox2D(size, spawnDensity)
  const points = []

  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const tx = nx === 1 ? 0.5 : x / (nx - 1)
      const ty = ny === 1 ? 0.5 : y / (ny - 1)
      const px = (tx - 0.5) * size[0] + centre[0]
      const py = (ty - 0.5) * size[1] + centre[1]
      points.push([px, py])
    }
  }

  return points
}

function initParticles() {
  const rng = createRng(42)
  const points = []
  const velocities = []
  const spawn = sim2dConfig.spawner

  for (const region of spawn.spawnRegions) {
    const regionPoints = spawnInRegion(region, spawn.spawnDensity)
    for (const point of regionPoints) {
      const angle = rng() * Math.PI * 2
      const dir = [Math.cos(angle), Math.sin(angle)]
      const jitter = (rng() - 0.5) * spawn.jitterStrength
      points.push([point[0] + dir[0] * jitter, point[1] + dir[1] * jitter])
      velocities.push([spawn.initialVelocity[0], spawn.initialVelocity[1]])
    }
  }

  state.positions = points
  state.velocities = velocities
}

function sampleGradient(stops, t) {
  if (stops.length === 0) return [1, 1, 1]
  if (t <= stops[0].t) return stops[0].color
  if (t >= stops[stops.length - 1].t) return stops[stops.length - 1].color
  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i]
    const b = stops[i + 1]
    if (t >= a.t && t <= b.t) {
      const u = (t - a.t) / (b.t - a.t)
      return [
        a.color[0] + (b.color[0] - a.color[0]) * u,
        a.color[1] + (b.color[1] - a.color[1]) * u,
        a.color[2] + (b.color[2] - a.color[2]) * u,
      ]
    }
  }
  return stops[0].color
}

function worldToScreen(pos, view, size) {
  const x = (pos[0] - view.left) / (view.right - view.left)
  const y = (pos[1] - view.bottom) / (view.top - view.bottom)
  return [x * size.width, (1 - y) * size.height]
}

function buildView(size) {
  const orthoSize = sim2dConfig.camera.orthographicSize
  const aspect = size.width / size.height
  return {
    left: -orthoSize * aspect,
    right: orthoSize * aspect,
    bottom: -orthoSize,
    top: orthoSize,
  }
}

function drawParticles() {
  const size = state.viewSize
  const view = buildView(size)
  const pixelsPerUnit = size.height / (view.top - view.bottom)
  const radius = sim2dConfig.render.particleScale * pixelsPerUnit

  for (let i = 0; i < state.positions.length; i++) {
    const pos = state.positions[i]
    const vel = state.velocities[i]
    const speed = Math.hypot(vel[0], vel[1])
    const t = Math.min(speed / sim2dConfig.render.velocityDisplayMax, 1)
    const color = sampleGradient(sim2dConfig.render.gradientStops, t)
    const [x, y] = worldToScreen(pos, view, size)

    ctx.beginPath()
    ctx.arc(x, y, radius, 0, Math.PI * 2)
    ctx.fillStyle = `rgb(${Math.round(color[0] * 255)}, ${Math.round(
      color[1] * 255
    )}, ${Math.round(color[2] * 255)})`
    ctx.fill()
  }
}

function drawOverlay() {
  const size = state.viewSize
  const view = buildView(size)
  const bounds = sim2dConfig.sim.boundsSize
  const halfBounds = [bounds[0] * 0.5, bounds[1] * 0.5]
  const bottomLeft = worldToScreen([-halfBounds[0], -halfBounds[1]], view, size)
  const topRight = worldToScreen([halfBounds[0], halfBounds[1]], view, size)
  const width = topRight[0] - bottomLeft[0]
  const height = bottomLeft[1] - topRight[1]

  ctx.strokeStyle = 'rgba(255,255,255,0.35)'
  ctx.lineWidth = 1
  ctx.strokeRect(bottomLeft[0], topRight[1], width, height)

  const origin = worldToScreen([0, 0], view, size)
  ctx.strokeStyle = 'rgba(255,255,255,0.25)'
  ctx.beginPath()
  ctx.moveTo(0, origin[1])
  ctx.lineTo(size.width, origin[1])
  ctx.moveTo(origin[0], 0)
  ctx.lineTo(origin[0], size.height)
  ctx.stroke()

  ctx.fillStyle = 'rgba(255,255,255,0.8)'
  ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace'
  ctx.textBaseline = 'top'
  ctx.fillText(`particles: ${state.positions.length}`, 12, 12)
  ctx.fillText(`webgpu: ${state.gpuStatus}`, 12, 28)
}

function configureWebGPU() {
  if (!state.gpuDevice || !state.gpuContext || !state.gpuFormat) return
  state.gpuContext.configure({
    device: state.gpuDevice,
    format: state.gpuFormat,
    alphaMode: 'opaque',
  })
}

async function initWebGPU() {
  if (!('gpu' in navigator)) {
    state.gpuStatus = 'unsupported'
    return
  }

  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) {
    state.gpuStatus = 'no adapter'
    return
  }

  state.gpuDevice = await adapter.requestDevice()
  state.gpuContext = gpuCanvas.getContext('webgpu')
  state.gpuFormat = navigator.gpu.getPreferredCanvasFormat()
  configureWebGPU()
  state.gpuStatus = 'ready'
}

function resize() {
  const dpr = Math.max(1, window.devicePixelRatio || 1)
  const width = window.innerWidth
  const height = window.innerHeight
  gpuCanvas.width = Math.floor(width * dpr)
  gpuCanvas.height = Math.floor(height * dpr)
  gpuCanvas.style.width = `${width}px`
  gpuCanvas.style.height = `${height}px`
  canvas.width = Math.floor(width * dpr)
  canvas.height = Math.floor(height * dpr)
  canvas.style.width = `${width}px`
  canvas.style.height = `${height}px`
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  state.viewSize.width = width
  state.viewSize.height = height
  configureWebGPU()
}

function frame() {
  if (state.gpuDevice && state.gpuContext) {
    const encoder = state.gpuDevice.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: state.gpuContext.getCurrentTexture().createView(),
          clearValue: state.gpuClearColor,
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    })
    pass.end()
    state.gpuDevice.queue.submit([encoder.finish()])
  }
  ctx.fillStyle = sim2dConfig.render.clearColor
  ctx.fillRect(0, 0, state.viewSize.width, state.viewSize.height)
  drawParticles()
  drawOverlay()
  requestAnimationFrame(frame)
}

window.addEventListener('resize', resize)
initParticles()
initWebGPU()
resize()
frame()
