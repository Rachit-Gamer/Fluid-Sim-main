using UnityEngine;
using Unity.Mathematics;
using System;
using UnityEngine.Experimental.Rendering;
using System.Collections;
using UnityEngine.Rendering;

public class Simulation2DOptimized : MonoBehaviour
{
    public event System.Action SimulationStepCompleted;

    [Header("Simulation Settings")]
    public float timeScale = 1;
    public bool fixedTimeStep;
    public int iterationsPerFrame;
    public float gravity;
    [Range(0, 1)] public float collisionDamping = 0.95f;
    public float smoothingRadius = 2;
    public float targetDensity;
    public float pressureMultiplier;
    public float nearPressureMultiplier;
    public float viscosityStrength;
    public Vector2 boundsSize;
    public Vector2 obstacleSize;
    public Vector2 obstacleCentre;

    [Header("Interaction Settings")]
    public float interactionRadius;
    public float interactionStrength;

    [Header("References")]
    public ComputeShader compute;
    public ParticleSpawner spawner;
    public ParticleDisplay2D display;

    public Simulation2D sim;
    // Buffers
    public ComputeBuffer positionBuffer { get; private set; }
    public ComputeBuffer velocityBuffer { get; private set; }
    public ComputeBuffer densityBuffer { get; private set; }
    ComputeBuffer predictedPositionBuffer;
    ComputeBuffer spatialIndices;
    ComputeBuffer spatialOffsets;
    GPUSort gpuSort;

    // Kernel IDs
    const int externalForcesKernel = 0;
    const int spatialHashKernel = 1;
    const int densityKernel = 2;
    const int pressureKernel = 3;
    const int viscosityKernel = 4;
    const int updatePositionKernel = 5;

    // State
    bool isPaused;
    ParticleSpawner.ParticleSpawnData spawnData;
    bool pauseNextFrame;

    public int numParticles { get; private set; }

    private float _timer = 0.0f;
    private float _inputCheckInterval = 0.1f;

    void Start()
    {
        Debug.Log("Controls: Space = Play/Pause, R = Reset, LMB = Attract, RMB = Repel");

        float deltaTime = 1 / 60f;
        Time.fixedDeltaTime = deltaTime;

        spawnData = spawner.GetSpawnData();
        numParticles = spawnData.positions.Length;

        // Create buffers
        positionBuffer = new ComputeBuffer(numParticles, sizeof(float) * 2, ComputeBufferType.Structured, ComputeBufferMode.Dynamic);
        predictedPositionBuffer = new ComputeBuffer(numParticles, sizeof(float) * 2, ComputeBufferType.Structured, ComputeBufferMode.Dynamic);
        velocityBuffer = new ComputeBuffer(numParticles, sizeof(float) * 2, ComputeBufferType.Structured, ComputeBufferMode.Dynamic);
        densityBuffer = new ComputeBuffer(numParticles, sizeof(float) * 2, ComputeBufferType.Structured, ComputeBufferMode.Dynamic);
        spatialIndices = new ComputeBuffer(numParticles, sizeof(uint) * 3, ComputeBufferType.Structured, ComputeBufferMode.Dynamic);
        spatialOffsets = new ComputeBuffer(numParticles, sizeof(uint), ComputeBufferType.Structured, ComputeBufferMode.Dynamic);


        // Set buffer data
        SetInitialBufferData(spawnData);

        // Init compute
        ComputeHelper.SetBuffer(compute, positionBuffer, "Positions", externalForcesKernel, updatePositionKernel);
        ComputeHelper.SetBuffer(compute, predictedPositionBuffer, "PredictedPositions", externalForcesKernel, spatialHashKernel, densityKernel, pressureKernel, viscosityKernel);
        ComputeHelper.SetBuffer(compute, spatialIndices, "SpatialIndices", spatialHashKernel, densityKernel, pressureKernel, viscosityKernel);
        ComputeHelper.SetBuffer(compute, spatialOffsets, "SpatialOffsets", spatialHashKernel, densityKernel, pressureKernel, viscosityKernel);
        ComputeHelper.SetBuffer(compute, densityBuffer, "Densities", densityKernel, pressureKernel, viscosityKernel);
        ComputeHelper.SetBuffer(compute, velocityBuffer, "Velocities", externalForcesKernel, pressureKernel, viscosityKernel, updatePositionKernel);

        compute.SetInt("numParticles", numParticles);

        gpuSort = new();
        gpuSort.SetBuffers(spatialIndices, spatialOffsets);

        // Init display
        display.Init(sim);
    }

    void FixedUpdate()
    {
        if (fixedTimeStep)
        {
            RunSimulationFrame(Time.fixedDeltaTime);
        }
    }

    void Update()
    {
        // Run simulation if not in fixed timestep mode
        // (skip running for first few frames as deltaTime can be disproportionally large)
        if (!fixedTimeStep && Time.frameCount > 10)
        {
            RunSimulationFrame(Time.deltaTime);
        }

        _timer += Time.deltaTime;
        if (_timer >= _inputCheckInterval)
        {
            _timer -= _inputCheckInterval;
            HandleInput();
        }

        if (pauseNextFrame)
        {
            isPaused = true;
            pauseNextFrame = false;
        }
    }

    void RunSimulationFrame(float frameTime)
    {
        if (!isPaused)
        {
            float timeStep = frameTime / iterationsPerFrame * timeScale;

            UpdateSettings(timeStep);

            for (int i = 0; i < iterationsPerFrame; i++)
            {
                RunSimulationStep();
                SimulationStepCompleted?.Invoke();
            }
        }
    }

    void RunSimulationStep()
    {
        ComputeHelper.Dispatch(compute, numParticles, kernelIndex: externalForcesKernel);
        ComputeHelper.Dispatch(compute, numParticles, kernelIndex: spatialHashKernel);
        gpuSort.SortAndCalculateOffsets();
        ComputeHelper.Dispatch(compute, numParticles, kernelIndex: densityKernel);
        ComputeHelper.Dispatch(compute, numParticles, kernelIndex: pressureKernel);
        ComputeHelper.Dispatch(compute, numParticles, kernelIndex: viscosityKernel);
        ComputeHelper.Dispatch(compute, numParticles, kernelIndex: updatePositionKernel);
    }

    void UpdateSettings(float deltaTime)
    {
        compute.SetFloat("deltaTime", deltaTime);
        compute.SetFloat("gravity", gravity);
        compute.SetFloat("collisionDamping", collisionDamping);
        compute.SetFloat("smoothingRadius", smoothingRadius);
        compute.SetFloat("targetDensity", targetDensity);
        compute.SetFloat("pressureMultiplier", pressureMultiplier);
        compute.SetFloat("nearPressureMultiplier", nearPressureMultiplier);
        compute.SetFloat("viscosityStrength", viscosityStrength);
        compute.SetVector("boundsSize", boundsSize);
        compute.SetVector("obstacleSize", obstacleSize);
        compute.SetVector("obstacleCentre", obstacleCentre);

        compute.SetFloat("Poly6ScalingFactor", 4 / (Mathf.PI * Mathf.Pow(smoothingRadius, 8)));
        compute.SetFloat("SpikyPow3ScalingFactor", 10 / (Mathf.PI * Mathf.Pow(smoothingRadius, 5)));
        compute.SetFloat("SpikyPow2ScalingFactor", 6 / (Mathf.PI * Mathf.Pow(smoothingRadius, 4)));
        compute.SetFloat("SpikyPow3DerivativeScalingFactor", 30 / (Mathf.Pow(smoothingRadius, 5) * Mathf.PI));
        compute.SetFloat("SpikyPow2DerivativeScalingFactor", 12 / (Mathf.Pow(smoothingRadius, 4) * Mathf.PI));

        // Mouse interaction settings:
        Vector2 mousePos = Camera.main.ScreenToWorldPoint(Input.mousePosition);
        bool isPullInteraction = Input.GetMouseButton(0);
        bool isPushInteraction = Input.GetMouseButton(1);
        float currInteractStrength = 0;
        if (isPushInteraction || isPullInteraction)
        {
            currInteractStrength = isPushInteraction ? -interactionStrength : interactionStrength;
        }

        compute.SetVector("interactionInputPoint", mousePos);
        compute.SetFloat("interactionInputStrength", currInteractStrength);
        compute.SetFloat("interactionInputRadius", interactionRadius);
    }

    IEnumerator SetInitialBufferData(ParticleSpawner.ParticleSpawnData spawnData)
    {
        float2[] allPoints = new float2[spawnData.positions.Length];
        System.Array.Copy(spawnData.positions, allPoints, spawnData.positions.Length);

        // Write data to the ComputeBuffers.
        positionBuffer.SetData(allPoints);
        predictedPositionBuffer.SetData(allPoints);
        velocityBuffer.SetData(spawnData.velocities);

        // Create readback requests for the ComputeBuffers.
        AsyncGPUReadbackRequest positionReadbackRequest = AsyncGPUReadback.Request(positionBuffer, 0, sizeof(float) * 2 * numParticles);
        AsyncGPUReadbackRequest predictedPositionReadbackRequest = AsyncGPUReadback.Request(predictedPositionBuffer, 0, sizeof(float) * 2 * numParticles);
        AsyncGPUReadbackRequest velocityReadbackRequest = AsyncGPUReadback.Request(velocityBuffer, 0, sizeof(float) * 2 * numParticles);

        // Wait for the readback requests to complete.
        while (!positionReadbackRequest.done || !predictedPositionReadbackRequest.done || !velocityReadbackRequest.done)
        {
            yield return null;
        }

        // Get the data from the readback requests.
        float[] positionData = new float[numParticles * 2];
        float[] predictedPositionData = new float[numParticles * 2];
        float[] velocityData = new float[numParticles * 2];
        positionBuffer.GetData(positionData);
        predictedPositionBuffer.GetData(predictedPositionData);
        velocityBuffer.GetData(velocityData);

        /* Release the readback requests.
        positionReadbackRequest.Release();
        predictedPositionReadbackRequest.Release();
        velocityReadbackRequest.Release();
        */


    }


    void HandleInput()
    {
        if (Input.GetKey(KeyCode.Space))
        {
            isPaused = !isPaused;
        }
        if (Input.GetKey(KeyCode.RightArrow))
        {
            isPaused = false;
            pauseNextFrame = true;
        }

        if (Input.GetKey(KeyCode.R))
        {
            isPaused = true;
            // Reset positions, the run single frame to get density etc (for debug purposes) and then reset positions again
            SetInitialBufferData(spawnData);
            RunSimulationStep();
            SetInitialBufferData(spawnData);
        }
    }

    void OnDestroy()
    {
        positionBuffer.Dispose();
        predictedPositionBuffer.Dispose();
        velocityBuffer.Dispose();
        densityBuffer.Dispose();
        spatialIndices.Dispose();
        spatialOffsets.Dispose();
    }

    void OnDrawGizmos()
    {
        Gizmos.color = new Color(0, 1, 0, 0.4f);
        Gizmos.DrawWireCube(Vector2.zero, boundsSize);
        Gizmos.DrawWireCube(obstacleCentre, obstacleSize);

        if (Application.isPlaying)
        {
            Vector2 mousePos = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            bool isPullInteraction = Input.GetMouseButton(0);
            bool isPushInteraction = Input.GetMouseButton(1);
            bool isInteracting = isPullInteraction || isPushInteraction;
            if (isInteracting)
            {
                Gizmos.color = isPullInteraction ? Color.green : Color.red;
                Gizmos.DrawWireSphere(mousePos, interactionRadius);
            }
        }
    }
}

