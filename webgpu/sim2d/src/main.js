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
  gpuPipeline: null,
  gpuBindGroup: null,
  gpuPositionBuffer: null,
  gpuUniformBuffer: null,
  gpuVertexBuffer: null,
  gpuComputePipeline: null,
  gpuComputeBindGroup: null,
  gpuVelocityBuffer: null,
  gpuComputeUniformBuffer: null,
  gpuError: '',
  useGpu: true,
  gpuCheckPending: true,
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
  if (state.gpuError) {
    ctx.fillText(state.gpuError, 12, 44)
  }
}

function initGpuResources() {
  const device = state.gpuDevice
  if (!device) return

  device.pushErrorScope('validation')

  const quadVertices = new Float32Array([
    -1, -1,
    1, -1,
    1, 1,
    -1, -1,
    1, 1,
    -1, 1,
  ])
  state.gpuVertexBuffer = device.createBuffer({
    size: quadVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(state.gpuVertexBuffer, 0, quadVertices)

  const positionsData = new Float32Array(state.positions.length * 2)
  for (let i = 0; i < state.positions.length; i++) {
    positionsData[i * 2] = state.positions[i][0]
    positionsData[i * 2 + 1] = state.positions[i][1]
  }
  state.gpuPositionBuffer = device.createBuffer({
    size: positionsData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(state.gpuPositionBuffer, 0, positionsData)

  state.gpuUniformBuffer = device.createBuffer({
    size: 16 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const renderShader = device.createShaderModule({
    code: `
struct SimUniforms {
  view: vec4<f32>,
  color: vec4<f32>,
  scale: f32,
  _pad: vec3<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: SimUniforms;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
};

@vertex
fn vsMain(@location(0) offset: vec2<f32>, @location(1) instancePos: vec2<f32>) -> VertexOut {
  let pos = instancePos;
  let world = pos + offset * uniforms.scale;
  let clipX = (world.x - uniforms.view.x) / (uniforms.view.y - uniforms.view.x) * 2.0 - 1.0;
  let clipY = (world.y - uniforms.view.z) / (uniforms.view.w - uniforms.view.z) * 2.0 - 1.0;
  var out: VertexOut;
  out.position = vec4<f32>(clipX, clipY, 0.0, 1.0);
  out.local = offset;
  return out;
}

@fragment
fn fsMain(in: VertexOut) -> @location(0) vec4<f32> {
  let dist = length(in.local);
  let aa = fwidth(dist);
  let alpha = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, dist);
  return vec4<f32>(uniforms.color.rgb, uniforms.color.a * alpha);
}
`,
  })

  state.gpuPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: renderShader,
      entryPoint: 'vsMain',
      buffers: [
        {
          arrayStride: 8,
          attributes: [{ shaderLocation: 0, format: 'float32x2', offset: 0 }],
        },
        {
          arrayStride: 8,
          stepMode: 'instance',
          attributes: [{ shaderLocation: 1, format: 'float32x2', offset: 0 }],
        },
      ],
    },
    fragment: {
      module: renderShader,
      entryPoint: 'fsMain',
      targets: [
        {
          format: state.gpuFormat,
          blend: {
            color: {
              srcFactor: 'src-alpha',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
            alpha: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
          },
        },
      ],
    },
    primitive: {
      topology: 'triangle-list',
      cullMode: 'none',
    },
  })

  state.gpuBindGroup = device.createBindGroup({
    layout: state.gpuPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: state.gpuUniformBuffer } },
    ],
  })

  const velocityData = new Float32Array(state.positions.length * 2)
  state.gpuVelocityBuffer = device.createBuffer({
    size: velocityData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(state.gpuVelocityBuffer, 0, velocityData)

  state.gpuComputeUniformBuffer = device.createBuffer({
    size: 4 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const computeShader = device.createShaderModule({
    code: `
struct ComputeUniforms {
  deltaTime: f32,
  gravity: f32,
  _pad: vec2<f32>,
};

@group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> uniforms: ComputeUniforms;

@compute @workgroup_size(64)
fn csMain(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&positions)) { return; }
  var v = velocities[i];
  v = v + vec2<f32>(0.0, uniforms.gravity) * uniforms.deltaTime;
  velocities[i] = v;
  positions[i] = positions[i] + v * uniforms.deltaTime;
}
`,
  })

  state.gpuComputePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: computeShader,
      entryPoint: 'csMain',
    },
  })

  state.gpuComputeBindGroup = device.createBindGroup({
    layout: state.gpuComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: state.gpuPositionBuffer } },
      { binding: 1, resource: { buffer: state.gpuVelocityBuffer } },
      { binding: 2, resource: { buffer: state.gpuComputeUniformBuffer } },
    ],
  })

  device.popErrorScope().then((error) => {
    if (error) {
      state.gpuStatus = 'error'
      state.gpuError = error.message
      state.useGpu = false
    }
  })
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
  state.gpuDevice.addEventListener('uncapturederror', (event) => {
    state.gpuStatus = 'error'
    state.gpuError = event.error?.message ?? 'uncaptured error'
    state.useGpu = false
  })
  state.gpuDevice.lost.then((info) => {
    state.gpuStatus = 'lost'
    state.gpuError = info?.message ?? 'device lost'
    state.useGpu = false
  })
  state.gpuContext = gpuCanvas.getContext('webgpu')
  state.gpuFormat = navigator.gpu.getPreferredCanvasFormat()
  configureWebGPU()
  initGpuResources()
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

function updateGpuUniforms() {
  if (!state.gpuDevice || !state.gpuUniformBuffer) return
  const view = buildView(state.viewSize)
  const color = sim2dConfig.render.gradientStops[0]?.color ?? [1, 1, 1]
  const data = new Float32Array(16)
  data[0] = view.left
  data[1] = view.right
  data[2] = view.bottom
  data[3] = view.top
  data[4] = color[0]
  data[5] = color[1]
  data[6] = color[2]
  data[7] = 1
  data[8] = sim2dConfig.render.particleScale
  data[9] = 0
  data[10] = 0
  data[11] = 0
  state.gpuDevice.queue.writeBuffer(state.gpuUniformBuffer, 0, data)
}

function updateComputeUniforms() {
  if (!state.gpuDevice || !state.gpuComputeUniformBuffer) return
  const data = new Float32Array(4)
  data[0] = 1 / 60
  data[1] = sim2dConfig.sim.gravity
  state.gpuDevice.queue.writeBuffer(state.gpuComputeUniformBuffer, 0, data)
}

function frame() {
  const useGpu = state.useGpu && state.gpuDevice && state.gpuContext && state.gpuPipeline
  if (useGpu) {
    updateGpuUniforms()
    updateComputeUniforms()
    if (state.gpuCheckPending) {
      state.gpuDevice.pushErrorScope('validation')
    }
    const encoder = state.gpuDevice.createCommandEncoder()
    if (state.gpuComputePipeline && state.gpuComputeBindGroup) {
      const computePass = encoder.beginComputePass()
      computePass.setPipeline(state.gpuComputePipeline)
      computePass.setBindGroup(0, state.gpuComputeBindGroup)
      const workgroupCount = Math.ceil(state.positions.length / 64)
      computePass.dispatchWorkgroups(workgroupCount)
      computePass.end()
    }
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
    pass.setPipeline(state.gpuPipeline)
    pass.setBindGroup(0, state.gpuBindGroup)
    pass.setVertexBuffer(0, state.gpuVertexBuffer)
    pass.setVertexBuffer(1, state.gpuPositionBuffer)
    pass.draw(6, state.positions.length, 0, 0)
    pass.end()
    state.gpuDevice.queue.submit([encoder.finish()])
    if (state.gpuCheckPending) {
      state.gpuDevice.popErrorScope().then((error) => {
        if (error) {
          state.gpuStatus = 'error'
          state.gpuError = error.message
          state.useGpu = false
        }
      })
      state.gpuCheckPending = false
    }
  }
  if (useGpu) {
    ctx.clearRect(0, 0, state.viewSize.width, state.viewSize.height)
  } else {
    ctx.fillStyle = sim2dConfig.render.clearColor
    ctx.fillRect(0, 0, state.viewSize.width, state.viewSize.height)
    drawParticles()
  }
  drawOverlay()
  requestAnimationFrame(frame)
}

window.addEventListener('resize', resize)
initParticles()
initWebGPU()
resize()
frame()
