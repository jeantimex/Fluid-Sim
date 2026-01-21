// Spawner settings from Unity scene (Test A 2D)
const SPAWN_CONFIG = {
  spawnDensity: 159,
  initialVelocity: [0, 0],
  jitterStr: 0.03,
  spawnRegions: [
    {
      position: [0, 0.66],
      size: [6.42, 4.39],
    },
  ],
};

// Seeded random number generator (matching Unity's Random(42))
function createRng(seed) {
  let state = seed;
  return function () {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

// Calculate spawn count per axis (matching Unity's CalculateSpawnCountPerAxisBox2D)
function calculateSpawnCountPerAxis(size, spawnDensity) {
  const area = size[0] * size[1];
  const targetTotal = Math.ceil(area * spawnDensity);

  const lenSum = size[0] + size[1];
  const tx = size[0] / lenSum;
  const ty = size[1] / lenSum;
  const m = Math.sqrt(targetTotal / (tx * ty));
  const nx = Math.ceil(tx * m);
  const ny = Math.ceil(ty * m);

  return [nx, ny];
}

// Spawn particles in a region (matching Unity's SpawnInRegion)
function spawnInRegion(region, spawnDensity) {
  const centre = region.position;
  const size = region.size;
  const [nx, ny] = calculateSpawnCountPerAxis(size, spawnDensity);
  const points = [];

  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const tx = nx > 1 ? x / (nx - 1) : 0.5;
      const ty = ny > 1 ? y / (ny - 1) : 0.5;

      const px = (tx - 0.5) * size[0] + centre[0];
      const py = (ty - 0.5) * size[1] + centre[1];
      points.push([px, py]);
    }
  }

  return points;
}

// Generate spawn data (matching Unity's GetSpawnData)
export function getSpawnData(config = SPAWN_CONFIG) {
  const rng = createRng(42);

  const allPositions = [];
  const allVelocities = [];

  for (const region of config.spawnRegions) {
    const points = spawnInRegion(region, config.spawnDensity);

    for (const point of points) {
      // Add jitter
      const angle = rng() * Math.PI * 2;
      const dir = [Math.cos(angle), Math.sin(angle)];
      const jitterAmount = config.jitterStr * (rng() - 0.5);

      allPositions.push([
        point[0] + dir[0] * jitterAmount,
        point[1] + dir[1] * jitterAmount,
      ]);
      allVelocities.push([...config.initialVelocity]);
    }
  }

  return {
    positions: allPositions,
    velocities: allVelocities,
    numParticles: allPositions.length,
  };
}

// Convert positions array to Float32Array for GPU buffer
export function positionsToFloat32Array(positions) {
  const data = new Float32Array(positions.length * 2);
  for (let i = 0; i < positions.length; i++) {
    data[i * 2] = positions[i][0];
    data[i * 2 + 1] = positions[i][1];
  }
  return data;
}

// Convert velocities array to Float32Array for GPU buffer
export function velocitiesToFloat32Array(velocities) {
  const data = new Float32Array(velocities.length * 2);
  for (let i = 0; i < velocities.length; i++) {
    data[i * 2] = velocities[i][0];
    data[i * 2 + 1] = velocities[i][1];
  }
  return data;
}
