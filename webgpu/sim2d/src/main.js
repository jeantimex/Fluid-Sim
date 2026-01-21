import { getSpawnData, positionsToFloat32Array } from './spawner.js';

const canvas = document.getElementById('canvas');
const status = document.getElementById('status');

// Particle settings from Unity scene
const PARTICLE_SCALE = 0.08;

// Simulation bounds from Unity scene (world units)
const BOUNDS_WIDTH = 17.1;
const BOUNDS_HEIGHT = 9.3;

// Canvas size - maintain aspect ratio of bounds
const SCALE = 80; // pixels per world unit
const CANVAS_WIDTH = Math.floor(BOUNDS_WIDTH * SCALE);  // ~1368
const CANVAS_HEIGHT = Math.floor(BOUNDS_HEIGHT * SCALE); // ~744

canvas.width = CANVAS_WIDTH;
canvas.height = CANVAS_HEIGHT;

// World coordinate system (centered at origin)
const WORLD_MIN_X = -BOUNDS_WIDTH / 2;  // -8.55
const WORLD_MAX_X = BOUNDS_WIDTH / 2;   //  8.55
const WORLD_MIN_Y = -BOUNDS_HEIGHT / 2; // -4.65
const WORLD_MAX_Y = BOUNDS_HEIGHT / 2;  //  4.65

async function initWebGPU() {
  // Check WebGPU support
  if (!navigator.gpu) {
    status.textContent = 'WebGPU not supported in this browser';
    throw new Error('WebGPU not supported');
  }

  // Request adapter
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    status.textContent = 'No GPU adapter found';
    throw new Error('No GPU adapter found');
  }

  // Request device
  const device = await adapter.requestDevice();

  // Configure canvas context
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format,
    alphaMode: 'premultiplied',
  });

  status.textContent = 'WebGPU initialized';

  return { device, context, format };
}

// Create a simple line shader for drawing bounds
function createBoundsShader(device, format) {
  const shaderCode = /* wgsl */`
    struct Uniforms {
      worldToClip: mat4x4f,
    }

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) color: vec4f,
    }

    @vertex
    fn vertexMain(@location(0) pos: vec2f, @location(1) color: vec4f) -> VertexOutput {
      var out: VertexOutput;
      out.position = uniforms.worldToClip * vec4f(pos, 0.0, 1.0);
      out.color = color;
      return out;
    }

    @fragment
    fn fragmentMain(@location(0) color: vec4f) -> @location(0) vec4f {
      return color;
    }
  `;

  const shaderModule = device.createShaderModule({ code: shaderCode });

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: shaderModule,
      entryPoint: 'vertexMain',
      buffers: [
        {
          arrayStride: 24, // 2 floats pos + 4 floats color = 6 * 4 bytes
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x2' },  // position
            { shaderLocation: 1, offset: 8, format: 'float32x4' },  // color
          ],
        },
      ],
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fragmentMain',
      targets: [{ format }],
    },
    primitive: {
      topology: 'line-list',
    },
  });

  return pipeline;
}

// Create particle shader for instanced circle rendering
function createParticleShader(device, format) {
  const shaderCode = /* wgsl */`
    struct Uniforms {
      worldToClip: mat4x4f,
      particleScale: f32,
    }

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var<storage, read> positions: array<vec2f>;

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) localPos: vec2f,
    }

    @vertex
    fn vertexMain(
      @builtin(vertex_index) vertexIndex: u32,
      @builtin(instance_index) instanceIndex: u32
    ) -> VertexOutput {
      // Quad vertices (2 triangles forming a square)
      let quadVerts = array<vec2f, 6>(
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0),
        vec2f( 1.0,  1.0),
        vec2f(-1.0, -1.0),
        vec2f( 1.0,  1.0),
        vec2f(-1.0,  1.0),
      );

      let localPos = quadVerts[vertexIndex];
      let particlePos = positions[instanceIndex];
      let worldPos = particlePos + localPos * uniforms.particleScale;

      var out: VertexOutput;
      out.position = uniforms.worldToClip * vec4f(worldPos, 0.0, 1.0);
      out.localPos = localPos;
      return out;
    }

    @fragment
    fn fragmentMain(@location(0) localPos: vec2f) -> @location(0) vec4f {
      // Draw circle: discard pixels outside radius
      let dist = length(localPos);
      if (dist > 1.0) {
        discard;
      }

      // Simple blue color for now (will add velocity coloring later)
      let color = vec3f(0.13, 0.34, 0.73); // Blue from Unity gradient
      return vec4f(color, 1.0);
    }
  `;

  const shaderModule = device.createShaderModule({ code: shaderCode });

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: shaderModule,
      entryPoint: 'vertexMain',
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fragmentMain',
      targets: [{ format }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  return pipeline;
}

// Create orthographic projection matrix (world coords to clip space)
function createOrthoMatrix(left, right, bottom, top) {
  // Maps [left,right] to [-1,1] and [bottom,top] to [-1,1]
  const sx = 2 / (right - left);
  const sy = 2 / (top - bottom);
  const tx = -(right + left) / (right - left);
  const ty = -(top + bottom) / (top - bottom);

  return new Float32Array([
    sx, 0,  0, 0,
    0,  sy, 0, 0,
    0,  0,  1, 0,
    tx, ty, 0, 1,
  ]);
}

async function main() {
  try {
    const { device, context, format } = await initWebGPU();

    // === Spawn particles ===
    const spawnData = getSpawnData();
    const numParticles = spawnData.numParticles;
    console.log(`Spawned ${numParticles} particles`);

    // Create position buffer for particles
    const positionData = positionsToFloat32Array(spawnData.positions);
    const positionBuffer = device.createBuffer({
      size: positionData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(positionBuffer, 0, positionData);

    // === Create pipelines ===
    const boundsPipeline = createBoundsShader(device, format);
    const particlePipeline = createParticleShader(device, format);

    // === Create uniform buffers ===
    // Bounds uniform (just projection matrix)
    const boundsUniformBuffer = device.createBuffer({
      size: 64, // 4x4 matrix
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Particle uniform (projection matrix + scale, padded to 16-byte alignment)
    const particleUniformBuffer = device.createBuffer({
      size: 80, // 64 (mat4) + 4 (f32) + 12 (padding) = 80
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Set orthographic projection
    const projMatrix = createOrthoMatrix(WORLD_MIN_X, WORLD_MAX_X, WORLD_MIN_Y, WORLD_MAX_Y);
    device.queue.writeBuffer(boundsUniformBuffer, 0, projMatrix);
    device.queue.writeBuffer(particleUniformBuffer, 0, projMatrix);
    device.queue.writeBuffer(particleUniformBuffer, 64, new Float32Array([PARTICLE_SCALE]));

    // === Create bind groups ===
    const boundsBindGroup = device.createBindGroup({
      layout: boundsPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: boundsUniformBuffer } }],
    });

    const particleBindGroup = device.createBindGroup({
      layout: particlePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: particleUniformBuffer } },
        { binding: 1, resource: { buffer: positionBuffer } },
      ],
    });

    // === Create bounds vertex buffer ===
    const green = [0.0, 1.0, 0.0, 0.4];
    const hw = BOUNDS_WIDTH / 2;
    const hh = BOUNDS_HEIGHT / 2;

    const boundsVertices = new Float32Array([
      -hw, -hh, ...green,
       hw, -hh, ...green,
       hw, -hh, ...green,
       hw,  hh, ...green,
       hw,  hh, ...green,
      -hw,  hh, ...green,
      -hw,  hh, ...green,
      -hw, -hh, ...green,
    ]);

    const boundsVertexBuffer = device.createBuffer({
      size: boundsVertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(boundsVertexBuffer, 0, boundsVertices);

    status.textContent = `Particles: ${numParticles} | Bounds: ${BOUNDS_WIDTH}x${BOUNDS_HEIGHT}`;

    // === Render loop ===
    function render() {
      const commandEncoder = device.createCommandEncoder();

      const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });

      // Draw particles (instanced)
      renderPass.setPipeline(particlePipeline);
      renderPass.setBindGroup(0, particleBindGroup);
      renderPass.draw(6, numParticles); // 6 vertices per quad, numParticles instances

      // Draw bounds outline
      renderPass.setPipeline(boundsPipeline);
      renderPass.setBindGroup(0, boundsBindGroup);
      renderPass.setVertexBuffer(0, boundsVertexBuffer);
      renderPass.draw(8);

      renderPass.end();

      device.queue.submit([commandEncoder.finish()]);
      requestAnimationFrame(render);
    }

    render();
  } catch (error) {
    console.error('Failed to initialize:', error);
  }
}

main();
