export const sim2dConfig = {
  sim: {
    timeScale: 1,
    maxTimestepFPS: 60,
    iterationsPerFrame: 3,
    gravity: -12,
    collisionDamping: 0.95,
    smoothingRadius: 0.35,
    targetDensity: 55,
    pressureMultiplier: 500,
    nearPressureMultiplier: 5,
    viscosityStrength: 0.03,
    boundsSize: [17.1, 9.3],
    obstacleSize: [0, 0],
    obstacleCenter: [0, 0],
    interactionRadius: 2,
    interactionStrength: 90,
  },
  spawner: {
    spawnDensity: 159,
    initialVelocity: [0, 0],
    jitterStrength: 0.03,
    spawnRegions: [
      {
        position: [0, 0.66],
        size: [6.42, 4.39],
      },
    ],
    expectedParticleCount: 4536,
  },
  render: {
    clearColor: '#000000',
    particleScale: 0.08,
    velocityDisplayMax: 6.5,
    gradientResolution: 64,
    gradientStops: [
      { t: 4064 / 65535, color: [0.13363299, 0.34235913, 0.7264151] },
      { t: 33191 / 65535, color: [0.2980392, 1, 0.56327766] },
      { t: 46738 / 65535, color: [1, 0.9309917, 0] },
      { t: 65535 / 65535, color: [0.96862745, 0.28555763, 0.031372573] },
    ],
  },
  camera: {
    orthographicSize: 5,
    position: [0, 0, -10],
  },
}
