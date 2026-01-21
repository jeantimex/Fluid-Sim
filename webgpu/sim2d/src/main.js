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
  gpuPredictedPositionBuffer: null,
  gpuUniformBuffer: null,
  gpuVertexBuffer: null,
  gpuComputePipeline: null,
  gpuUpdatePipeline: null,
  gpuComputeBindGroup: null,
  gpuVelocityBuffer: null,
  gpuComputeUniformBuffer: null,
  gpuSpatialKeysBuffer: null,
  gpuSpatialOffsetsBuffer: null,
  gpuSortedIndicesBuffer: null,
  gpuSpatialPipeline: null,
  gpuSpatialBindGroup: null,
  gpuSpatialUniformBuffer: null,
  gpuSpatialReady: false,
  gpuCountBuffer: null,
  gpuScanUniformBuffer: null,
  gpuScanGroupSumsBuffer: null,
  gpuScanGroupSumsBuffer2: null,
  gpuScanPipeline: null,
  gpuScanCombinePipeline: null,
  gpuScanReady: false,
  gpuScanCheckPending: true,
  gpuCountSortPipeline: null,
  gpuCountSortScatterPipeline: null,
  gpuCountSortCopyPipeline: null,
  gpuCountSortClearPipeline: null,
  gpuCountSortBindGroup: null,
  gpuCountSortReady: false,
  gpuCountSortTempItemsBuffer: null,
  gpuCountSortTempKeysBuffer: null,
  gpuOffsetsPipeline: null,
  gpuOffsetsInitPipeline: null,
  gpuOffsetsBindGroup: null,
  gpuOffsetsReady: false,
  gpuDensityBuffer: null,
  gpuDensityPipeline: null,
  gpuDensityBindGroup: null,
  gpuDensityReady: false,
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

  state.gpuPredictedPositionBuffer = device.createBuffer({
    size: positionsData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })

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

  state.gpuSpatialKeysBuffer = device.createBuffer({
    size: state.positions.length * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  state.gpuSpatialOffsetsBuffer = device.createBuffer({
    size: state.positions.length * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  state.gpuSortedIndicesBuffer = device.createBuffer({
    size: state.positions.length * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  state.gpuDensityBuffer = device.createBuffer({
    size: state.positions.length * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  state.gpuCountSortTempItemsBuffer = device.createBuffer({
    size: state.positions.length * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  state.gpuCountSortTempKeysBuffer = device.createBuffer({
    size: state.positions.length * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })

  state.gpuSpatialUniformBuffer = device.createBuffer({
    size: 8 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  state.gpuCountBuffer = device.createBuffer({
    size: state.positions.length * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(state.gpuCountBuffer, 0, new Uint32Array(state.positions.length))

  const itemsPerGroup = 512
  const numGroups = Math.ceil(state.positions.length / itemsPerGroup)
  const numGroups2 = Math.ceil(numGroups / itemsPerGroup)
  state.gpuScanGroupSumsBuffer = device.createBuffer({
    size: Math.max(1, numGroups) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  state.gpuScanGroupSumsBuffer2 = device.createBuffer({
    size: Math.max(1, numGroups2) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })

  state.gpuScanUniformBuffer = device.createBuffer({
    size: 8 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  state.gpuComputeUniformBuffer = device.createBuffer({
    size: 16 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const computeShader = device.createShaderModule({
    code: `
struct ComputeUniforms {
  simParams: vec4<f32>,
  boundsSize: vec2<f32>,
  _pad0: vec2<f32>,
  obstacleSize: vec2<f32>,
  obstacleCenter: vec2<f32>,
};

@group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> predicted: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> uniforms: ComputeUniforms;

@compute @workgroup_size(64)
fn externalForces(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&positions)) { return; }
  let deltaTime = uniforms.simParams.x;
  let gravity = uniforms.simParams.y;
  let predictionFactor = uniforms.simParams.w;
  var v = velocities[i];
  v = v + vec2<f32>(0.0, gravity) * deltaTime;
  velocities[i] = v;
  predicted[i] = positions[i] + v * predictionFactor;
}

@compute @workgroup_size(64)
fn updatePositions(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&positions)) { return; }
  let deltaTime = uniforms.simParams.x;
  let collisionDamping = uniforms.simParams.z;
  var v = velocities[i];
  var p = positions[i] + v * deltaTime;
  let halfSize = uniforms.boundsSize * 0.5;
  let edgeDst = halfSize - abs(p);
  if (edgeDst.x <= 0.0) {
    p.x = halfSize.x * sign(p.x);
    v.x = -v.x * collisionDamping;
  }
  if (edgeDst.y <= 0.0) {
    p.y = halfSize.y * sign(p.y);
    v.y = -v.y * collisionDamping;
  }
  let obstacleHalf = uniforms.obstacleSize * 0.5;
  let obstacleDst = obstacleHalf - abs(p - uniforms.obstacleCenter);
  if (obstacleDst.x >= 0.0 && obstacleDst.y >= 0.0) {
    if (obstacleDst.x < obstacleDst.y) {
      p.x = obstacleHalf.x * sign(p.x - uniforms.obstacleCenter.x) + uniforms.obstacleCenter.x;
      v.x = -v.x * collisionDamping;
    } else {
      p.y = obstacleHalf.y * sign(p.y - uniforms.obstacleCenter.y) + uniforms.obstacleCenter.y;
      v.y = -v.y * collisionDamping;
    }
  }
  positions[i] = p;
  velocities[i] = v;
}
`,
  })

  const computeBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  })

  const computePipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [computeBindGroupLayout],
  })

  state.gpuComputePipeline = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: {
      module: computeShader,
      entryPoint: 'externalForces',
    },
  })

  state.gpuUpdatePipeline = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: {
      module: computeShader,
      entryPoint: 'updatePositions',
    },
  })

  state.gpuComputeBindGroup = device.createBindGroup({
    layout: computeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: state.gpuPositionBuffer } },
      { binding: 1, resource: { buffer: state.gpuPredictedPositionBuffer } },
      { binding: 2, resource: { buffer: state.gpuVelocityBuffer } },
      { binding: 3, resource: { buffer: state.gpuComputeUniformBuffer } },
    ],
  })

  device.pushErrorScope('validation')

  const spatialShader = device.createShaderModule({
    code: `
struct SpatialUniforms {
  smoothingRadius: f32,
  numParticles: f32,
  _pad: vec2<f32>,
};

@group(0) @binding(0) var<storage, read> predicted: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> keys: array<u32>;
@group(0) @binding(2) var<uniform> uniforms: SpatialUniforms;

const hashK1: u32 = 15823u;
const hashK2: u32 = 9737333u;

fn getCell2D(position: vec2<f32>, radius: f32) -> vec2<i32> {
  return vec2<i32>(
    i32(floor(position.x / radius)),
    i32(floor(position.y / radius))
  );
}

fn hashCell2D(cell: vec2<i32>) -> u32 {
  let ux = u32(cell.x);
  let uy = u32(cell.y);
  let a = ux * hashK1;
  let b = uy * hashK2;
  return a + b;
}

@compute @workgroup_size(64)
fn updateSpatialHash(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (f32(i) >= uniforms.numParticles) { return; }
  let cell = getCell2D(predicted[i], uniforms.smoothingRadius);
  let hash = hashCell2D(cell);
  let count = u32(uniforms.numParticles);
  keys[i] = hash % count;
}
`,
  })

  const spatialBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  })

  const spatialPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [spatialBindGroupLayout],
  })

  state.gpuSpatialPipeline = device.createComputePipeline({
    layout: spatialPipelineLayout,
    compute: {
      module: spatialShader,
      entryPoint: 'updateSpatialHash',
    },
  })

  state.gpuSpatialBindGroup = device.createBindGroup({
    layout: spatialBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: state.gpuPredictedPositionBuffer } },
      { binding: 1, resource: { buffer: state.gpuSpatialKeysBuffer } },
      { binding: 2, resource: { buffer: state.gpuSpatialUniformBuffer } },
    ],
  })

  device.popErrorScope().then((error) => {
    if (error) {
      state.gpuStatus = 'error'
      state.gpuError = error.message
      state.gpuSpatialPipeline = null
      state.gpuSpatialBindGroup = null
      state.gpuSpatialReady = false
    } else {
      state.gpuSpatialReady = true
    }
  })

  device.pushErrorScope('validation')

  const scanShader = device.createShaderModule({
    code: `
struct ScanUniforms {
  itemCount: u32,
  _pad: vec3<u32>,
};

@group(0) @binding(0) var<storage, read_write> elements: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> groupSums: array<u32>;
@group(0) @binding(2) var<uniform> uniforms: ScanUniforms;

const GROUP_SIZE: u32 = 256u;
const ITEMS_PER_GROUP: u32 = 512u;

var<workgroup> temp: array<u32, 512>;

@compute @workgroup_size(256)
fn blockScan(
  @builtin(global_invocation_id) globalId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) groupId: vec3<u32>
) {
  let threadLocal = localId.x;
  let group = groupId.x;
  let globalA = globalId.x * 2u;
  let globalB = globalA + 1u;
  let localA = threadLocal * 2u;
  let localB = localA + 1u;

  let hasA = globalA < uniforms.itemCount;
  let hasB = globalB < uniforms.itemCount;

  temp[localA] = select(0u, atomicLoad(&elements[globalA]), hasA);
  temp[localB] = select(0u, atomicLoad(&elements[globalB]), hasB);

  var offset = 1u;
  for (var numActive = GROUP_SIZE; numActive > 0u; numActive = numActive / 2u) {
    workgroupBarrier();
    if (threadLocal < numActive) {
      let indexA = offset * (localA + 1u) - 1u;
      let indexB = offset * (localB + 1u) - 1u;
      temp[indexB] = temp[indexA] + temp[indexB];
    }
    offset = offset * 2u;
  }

  if (threadLocal == 0u) {
    groupSums[group] = temp[ITEMS_PER_GROUP - 1u];
    temp[ITEMS_PER_GROUP - 1u] = 0u;
  }

  for (var numActive = 1u; numActive <= GROUP_SIZE; numActive = numActive * 2u) {
    workgroupBarrier();
    offset = offset / 2u;
    if (threadLocal < numActive) {
      let indexA = offset * (localA + 1u) - 1u;
      let indexB = offset * (localB + 1u) - 1u;
      let sum = temp[indexA] + temp[indexB];
      temp[indexA] = temp[indexB];
      temp[indexB] = sum;
    }
  }

  workgroupBarrier();
  if (hasA) { atomicStore(&elements[globalA], temp[localA]); }
  if (hasB) { atomicStore(&elements[globalB], temp[localB]); }
}

@compute @workgroup_size(256)
fn blockCombine(
  @builtin(global_invocation_id) globalId: vec3<u32>,
  @builtin(workgroup_id) groupId: vec3<u32>
) {
  let globalA = globalId.x * 2u;
  let globalB = globalA + 1u;
  if (globalA < uniforms.itemCount) {
    let v = atomicLoad(&elements[globalA]);
    atomicStore(&elements[globalA], v + groupSums[groupId.x]);
  }
  if (globalB < uniforms.itemCount) {
    let v = atomicLoad(&elements[globalB]);
    atomicStore(&elements[globalB], v + groupSums[groupId.x]);
  }
}
`,
  })

  const scanBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  })

  const scanPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [scanBindGroupLayout],
  })

  state.gpuScanPipeline = device.createComputePipeline({
    layout: scanPipelineLayout,
    compute: {
      module: scanShader,
      entryPoint: 'blockScan',
    },
  })

  state.gpuScanCombinePipeline = device.createComputePipeline({
    layout: scanPipelineLayout,
    compute: {
      module: scanShader,
      entryPoint: 'blockCombine',
    },
  })

  device.popErrorScope().then((error) => {
    if (error) {
      state.gpuStatus = 'error'
      state.gpuError = error.message
      state.gpuScanPipeline = null
      state.gpuScanCombinePipeline = null
      state.gpuScanReady = false
    } else {
      state.gpuScanReady = true
    }
  })

  device.pushErrorScope('validation')

  const countSortShader = device.createShaderModule({
    code: `
struct CountSortUniforms {
  numInputs: u32,
  _pad: vec3<u32>,
};

@group(0) @binding(0) var<storage, read_write> inputItems: array<u32>;
@group(0) @binding(1) var<storage, read_write> inputKeys: array<u32>;
@group(0) @binding(2) var<storage, read_write> sortedItems: array<u32>;
@group(0) @binding(3) var<storage, read_write> sortedKeys: array<u32>;
@group(0) @binding(4) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> uniforms: CountSortUniforms;

const GROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn clearCounts(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= uniforms.numInputs) { return; }
  atomicStore(&counts[i], 0u);
  inputItems[i] = i;
}

@compute @workgroup_size(256)
fn calculateCounts(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= uniforms.numInputs) { return; }
  let key = inputKeys[i];
  atomicAdd(&counts[key], 1u);
}

@compute @workgroup_size(256)
fn scatterOutput(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= uniforms.numInputs) { return; }
  let key = inputKeys[i];
  let sortedIndex = atomicAdd(&counts[key], 1u);
  sortedItems[sortedIndex] = inputItems[i];
  sortedKeys[sortedIndex] = key;
}

@compute @workgroup_size(256)
fn copyBack(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= uniforms.numInputs) { return; }
  inputItems[i] = sortedItems[i];
  inputKeys[i] = sortedKeys[i];
}
`,
  })

  const countSortBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  })

  const countSortPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [countSortBindGroupLayout],
  })

  state.gpuCountSortClearPipeline = device.createComputePipeline({
    layout: countSortPipelineLayout,
    compute: {
      module: countSortShader,
      entryPoint: 'clearCounts',
    },
  })

  state.gpuCountSortPipeline = device.createComputePipeline({
    layout: countSortPipelineLayout,
    compute: {
      module: countSortShader,
      entryPoint: 'calculateCounts',
    },
  })

  state.gpuCountSortScatterPipeline = device.createComputePipeline({
    layout: countSortPipelineLayout,
    compute: {
      module: countSortShader,
      entryPoint: 'scatterOutput',
    },
  })

  state.gpuCountSortCopyPipeline = device.createComputePipeline({
    layout: countSortPipelineLayout,
    compute: {
      module: countSortShader,
      entryPoint: 'copyBack',
    },
  })

  state.gpuCountSortBindGroup = device.createBindGroup({
    layout: countSortBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: state.gpuSortedIndicesBuffer } },
      { binding: 1, resource: { buffer: state.gpuSpatialKeysBuffer } },
      { binding: 2, resource: { buffer: state.gpuCountSortTempItemsBuffer } },
      { binding: 3, resource: { buffer: state.gpuCountSortTempKeysBuffer } },
      { binding: 4, resource: { buffer: state.gpuCountBuffer } },
      { binding: 5, resource: { buffer: state.gpuScanUniformBuffer } },
    ],
  })

  device.popErrorScope().then((error) => {
    if (error) {
      state.gpuStatus = 'error'
      state.gpuError = error.message
      state.gpuCountSortReady = false
      state.gpuCountSortPipeline = null
      state.gpuCountSortScatterPipeline = null
      state.gpuCountSortCopyPipeline = null
      state.gpuCountSortClearPipeline = null
      state.gpuCountSortBindGroup = null
    } else {
      state.gpuCountSortReady = true
    }
  })

  device.pushErrorScope('validation')

  const offsetsShader = device.createShaderModule({
    code: `
struct OffsetsUniforms {
  numInputs: u32,
  _pad: vec3<u32>,
};

@group(0) @binding(0) var<storage, read> sortedKeys: array<u32>;
@group(0) @binding(1) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(2) var<uniform> uniforms: OffsetsUniforms;

@compute @workgroup_size(256)
fn initializeOffsets(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= uniforms.numInputs) { return; }
  offsets[i] = uniforms.numInputs;
}

@compute @workgroup_size(256)
fn calculateOffsets(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= uniforms.numInputs) { return; }
  let key = sortedKeys[i];
  let keyPrev = select(uniforms.numInputs, sortedKeys[i - 1u], i > 0u);
  if (key != keyPrev) {
    offsets[key] = i;
  }
}
`,
  })

  const offsetsBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  })

  const offsetsPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [offsetsBindGroupLayout],
  })

  state.gpuOffsetsInitPipeline = device.createComputePipeline({
    layout: offsetsPipelineLayout,
    compute: {
      module: offsetsShader,
      entryPoint: 'initializeOffsets',
    },
  })

  state.gpuOffsetsPipeline = device.createComputePipeline({
    layout: offsetsPipelineLayout,
    compute: {
      module: offsetsShader,
      entryPoint: 'calculateOffsets',
    },
  })

  state.gpuOffsetsBindGroup = device.createBindGroup({
    layout: offsetsBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: state.gpuSpatialKeysBuffer } },
      { binding: 1, resource: { buffer: state.gpuSpatialOffsetsBuffer } },
      { binding: 2, resource: { buffer: state.gpuScanUniformBuffer } },
    ],
  })

  device.popErrorScope().then((error) => {
    if (error) {
      state.gpuStatus = 'error'
      state.gpuError = error.message
      state.gpuOffsetsReady = false
      state.gpuOffsetsPipeline = null
      state.gpuOffsetsInitPipeline = null
      state.gpuOffsetsBindGroup = null
    } else {
      state.gpuOffsetsReady = true
    }
  })

  device.pushErrorScope('validation')

  const densityShader = device.createShaderModule({
    code: `
struct DensityUniforms {
  smoothingRadius: f32,
  numParticles: f32,
  spiky3Factor: f32,
  spiky2Factor: f32,
};

@group(0) @binding(0) var<storage, read> predicted: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> spatialKeys: array<u32>;
@group(0) @binding(2) var<storage, read> spatialOffsets: array<u32>;
@group(0) @binding(3) var<storage, read> sortedIndices: array<u32>;
@group(0) @binding(4) var<storage, read_write> densities: array<vec2<f32>>;
@group(0) @binding(5) var<uniform> uniforms: DensityUniforms;

const hashK1: u32 = 15823u;
const hashK2: u32 = 9737333u;

fn getCell2D(position: vec2<f32>, radius: f32) -> vec2<i32> {
  return vec2<i32>(
    i32(floor(position.x / radius)),
    i32(floor(position.y / radius))
  );
}

fn hashCell2D(cell: vec2<i32>) -> u32 {
  let ux = u32(cell.x);
  let uy = u32(cell.y);
  let a = ux * hashK1;
  let b = uy * hashK2;
  return a + b;
}

fn spikyPow2(dst: f32, radius: f32, factor: f32) -> f32 {
  if (dst < radius) {
    let v = radius - dst;
    return v * v * factor;
  }
  return 0.0;
}

fn spikyPow3(dst: f32, radius: f32, factor: f32) -> f32 {
  if (dst < radius) {
    let v = radius - dst;
    return v * v * v * factor;
  }
  return 0.0;
}

@compute @workgroup_size(64)
fn calculateDensities(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (f32(i) >= uniforms.numParticles) { return; }
  let pos = predicted[i];
  let originCell = getCell2D(pos, uniforms.smoothingRadius);
  let sqrRadius = uniforms.smoothingRadius * uniforms.smoothingRadius;
  var density = 0.0;
  var nearDensity = 0.0;

  for (var oy = -1; oy <= 1; oy = oy + 1) {
    for (var ox = -1; ox <= 1; ox = ox + 1) {
      let cell = originCell + vec2<i32>(ox, oy);
      let hash = hashCell2D(cell);
      let count = u32(uniforms.numParticles);
      let key = hash % count;
      var currIndex = spatialOffsets[key];
      loop {
        if (currIndex >= count) { break; }
        let neighbourIndex = currIndex;
        currIndex = currIndex + 1u;
        let neighbourKey = spatialKeys[neighbourIndex];
        if (neighbourKey != key) { break; }
        let neighbourParticle = sortedIndices[neighbourIndex];
        let neighbourPos = predicted[neighbourParticle];
        let offset = neighbourPos - pos;
        let sqrDst = dot(offset, offset);
        if (sqrDst > sqrRadius) { continue; }
        let dst = sqrt(sqrDst);
        density = density + spikyPow2(dst, uniforms.smoothingRadius, uniforms.spiky2Factor);
        nearDensity = nearDensity + spikyPow3(dst, uniforms.smoothingRadius, uniforms.spiky3Factor);
      }
    }
  }

  densities[i] = vec2<f32>(density, nearDensity);
}
`,
  })

  const densityBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  })

  const densityPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [densityBindGroupLayout],
  })

  state.gpuDensityPipeline = device.createComputePipeline({
    layout: densityPipelineLayout,
    compute: {
      module: densityShader,
      entryPoint: 'calculateDensities',
    },
  })

  state.gpuDensityBindGroup = device.createBindGroup({
    layout: densityBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: state.gpuPredictedPositionBuffer } },
      { binding: 1, resource: { buffer: state.gpuSpatialKeysBuffer } },
      { binding: 2, resource: { buffer: state.gpuSpatialOffsetsBuffer } },
      { binding: 3, resource: { buffer: state.gpuSortedIndicesBuffer } },
      { binding: 4, resource: { buffer: state.gpuDensityBuffer } },
      { binding: 5, resource: { buffer: state.gpuSpatialUniformBuffer } },
    ],
  })

  device.popErrorScope().then((error) => {
    if (error) {
      state.gpuStatus = 'error'
      state.gpuError = error.message
      state.gpuDensityReady = false
      state.gpuDensityPipeline = null
      state.gpuDensityBindGroup = null
    } else {
      state.gpuDensityReady = true
    }
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
  const data = new Float32Array(16)
  data[0] = 1 / 60
  data[1] = sim2dConfig.sim.gravity
  data[2] = sim2dConfig.sim.collisionDamping
  data[3] = 1 / 120
  data[4] = sim2dConfig.sim.boundsSize[0]
  data[5] = sim2dConfig.sim.boundsSize[1]
  data[6] = 0
  data[7] = 0
  data[8] = sim2dConfig.sim.obstacleSize[0]
  data[9] = sim2dConfig.sim.obstacleSize[1]
  data[10] = sim2dConfig.sim.obstacleCenter[0]
  data[11] = sim2dConfig.sim.obstacleCenter[1]
  state.gpuDevice.queue.writeBuffer(state.gpuComputeUniformBuffer, 0, data)
}

function updateSpatialUniforms() {
  if (!state.gpuDevice || !state.gpuSpatialUniformBuffer) return
  const data = new Float32Array(4)
  data[0] = sim2dConfig.sim.smoothingRadius
  data[1] = state.positions.length
  data[2] = 10 / (Math.PI * Math.pow(sim2dConfig.sim.smoothingRadius, 5))
  data[3] = 6 / (Math.PI * Math.pow(sim2dConfig.sim.smoothingRadius, 4))
  state.gpuDevice.queue.writeBuffer(state.gpuSpatialUniformBuffer, 0, data)
}

function updateScanUniforms(itemCount) {
  if (!state.gpuDevice || !state.gpuScanUniformBuffer) return
  const data = new Uint32Array(4)
  data[0] = itemCount
  state.gpuDevice.queue.writeBuffer(state.gpuScanUniformBuffer, 0, data)
}

function runScanPass(encoder, elementsBuffer, groupSumsBuffer, itemCount) {
  if (!state.gpuScanReady || !state.gpuScanPipeline || !state.gpuScanCombinePipeline) return
  const bindGroup = state.gpuDevice.createBindGroup({
    layout: state.gpuScanPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: elementsBuffer } },
      { binding: 1, resource: { buffer: groupSumsBuffer } },
      { binding: 2, resource: { buffer: state.gpuScanUniformBuffer } },
    ],
  })
  const workgroupCount = Math.ceil(itemCount / 512)
  const pass = encoder.beginComputePass()
  updateScanUniforms(itemCount)
  pass.setPipeline(state.gpuScanPipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(workgroupCount)
  pass.setPipeline(state.gpuScanCombinePipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(workgroupCount)
  pass.end()
}

function frame() {
  const useGpu = state.useGpu && state.gpuDevice && state.gpuContext && state.gpuPipeline
  if (useGpu) {
    updateGpuUniforms()
    updateComputeUniforms()
    updateSpatialUniforms()
    if (state.gpuCheckPending) {
      state.gpuDevice.pushErrorScope('validation')
    }
    const encoder = state.gpuDevice.createCommandEncoder()
    if (state.gpuScanCheckPending && state.gpuScanReady) {
      runScanPass(encoder, state.gpuCountBuffer, state.gpuScanGroupSumsBuffer, state.positions.length)
      state.gpuScanCheckPending = false
    }
    if (state.gpuComputePipeline && state.gpuComputeBindGroup) {
      const workgroupCount = Math.ceil(state.positions.length / 64)
      const countSortGroups = Math.ceil(state.positions.length / 256)

      const computePass = encoder.beginComputePass()
      computePass.setPipeline(state.gpuComputePipeline)
      computePass.setBindGroup(0, state.gpuComputeBindGroup)
      computePass.dispatchWorkgroups(workgroupCount)
      if (state.gpuSpatialReady && state.gpuSpatialPipeline && state.gpuSpatialBindGroup) {
        computePass.setPipeline(state.gpuSpatialPipeline)
        computePass.setBindGroup(0, state.gpuSpatialBindGroup)
        computePass.dispatchWorkgroups(workgroupCount)
      }
      if (state.gpuCountSortReady && state.gpuCountSortBindGroup) {
        updateScanUniforms(state.positions.length)
        computePass.setPipeline(state.gpuCountSortClearPipeline)
        computePass.setBindGroup(0, state.gpuCountSortBindGroup)
        computePass.dispatchWorkgroups(countSortGroups)

        computePass.setPipeline(state.gpuCountSortPipeline)
        computePass.setBindGroup(0, state.gpuCountSortBindGroup)
        computePass.dispatchWorkgroups(countSortGroups)
      }
      computePass.end()

      if (state.gpuCountSortReady && state.gpuCountSortBindGroup) {
        runScanPass(encoder, state.gpuCountBuffer, state.gpuScanGroupSumsBuffer, state.positions.length)
      }

      const computePass2 = encoder.beginComputePass()
      if (state.gpuCountSortReady && state.gpuCountSortBindGroup) {
        computePass2.setPipeline(state.gpuCountSortScatterPipeline)
        computePass2.setBindGroup(0, state.gpuCountSortBindGroup)
        computePass2.dispatchWorkgroups(countSortGroups)

        computePass2.setPipeline(state.gpuCountSortCopyPipeline)
        computePass2.setBindGroup(0, state.gpuCountSortBindGroup)
        computePass2.dispatchWorkgroups(countSortGroups)
      }
      if (state.gpuOffsetsReady && state.gpuOffsetsBindGroup) {
        updateScanUniforms(state.positions.length)
        computePass2.setPipeline(state.gpuOffsetsInitPipeline)
        computePass2.setBindGroup(0, state.gpuOffsetsBindGroup)
        computePass2.dispatchWorkgroups(countSortGroups)

        computePass2.setPipeline(state.gpuOffsetsPipeline)
        computePass2.setBindGroup(0, state.gpuOffsetsBindGroup)
        computePass2.dispatchWorkgroups(countSortGroups)
      }
      if (state.gpuDensityReady && state.gpuDensityBindGroup) {
        computePass2.setPipeline(state.gpuDensityPipeline)
        computePass2.setBindGroup(0, state.gpuDensityBindGroup)
        computePass2.dispatchWorkgroups(workgroupCount)
      }
      if (state.gpuUpdatePipeline) {
        computePass2.setPipeline(state.gpuUpdatePipeline)
        computePass2.setBindGroup(0, state.gpuComputeBindGroup)
        computePass2.dispatchWorkgroups(workgroupCount)
      }
      computePass2.end()
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
