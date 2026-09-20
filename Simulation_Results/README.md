# Simulation Results

Output of the Force-Aware RRT* planner's 50-scenario simulation bank (thesis
§6.5). Every result here is from simulation, not physical hardware — for the
real-rig hardware pilot, see `Grasp_Validation/hardware_validation/` and
`Experiments/Grasp_Validation/` instead.

```
Simulation_Results/
├── Summary/
│   └── simulation_bank_summary.png   Aggregate view across all 50 runs:
│                                      load distribution, path length vs.
│                                      mass, runtime vs. tree size, obstacle
│                                      layout usage, waypoint distribution
└── Scenario_Bank/
    └── scenario_NN/                  One folder per successful run
        ├── start.png                 Arm and obstacles at the start config
        ├── tree.png                  RRT* search tree + the solution path
        ├── end.png                   Arm at the goal config
        └── trajectory.mp4            Animated playback of the planned motion
```

50 scenario folders, numbered by their original run ID. The numbering has
gaps (no `scenario_02`, `scenario_05`, etc.) — those IDs were among the ten
attempts that failed to find a solution within the iteration budget and were
excluded, out of 60 total attempts. The gaps are original run numbers, not
missing data.

Each scenario's title bar states its obstacle layout and object mass (e.g.
`Scenario 09 | A_baseline | m = 0.60 kg`), matching the aggregate table in
thesis §6.5.2.
