import io, contextlib, time, sys, random
import numpy as np, thesis_config as cfg
from RRTStar import RRTStar
OBST=[{'type':'circle','center':np.array([0.6,0.4]),'radius':0.07},
      {'type':'rectangle','bottom_left':np.array([0.8,0.7]),'width':0.1,'height':0.3}]
GOAL=[float(x) for x in sys.argv[1].split(',')]; ITERS=int(sys.argv[2]); N=int(sys.argv[3])
ok=[]; t0=time.time()
for k in range(N):
    np.random.seed(300+k); random.seed(300+k)
    buf=io.StringIO(); path=None
    with contextlib.redirect_stdout(buf):
        p=RRTStar(start_config=cfg.start_config, goal_config=GOAL,
            joint_limits=cfg.joint_limits, joint_vel_limit=cfg.joint_vel_limit,
            joint_acc_limit=cfg.joint_acc_limit, link_lengths=cfg.link_lengths,
            link_radii=cfg.link_radii, safety_margin=cfg.safety_margin, obstacles=OBST,
            S_a=cfg.S_a, S_v=cfg.S_v, mass=cfg.link_mass, inertia=cfg.link_inertia,
            gripper_param=cfg.gripper_param, object_param=cfg.object_param,
            adhesive_param=cfg.adhesive_param, angular_energy_friction=cfg.angular_energy_friction,
            step_size=0.05, goal_tolerance=0.05, max_iterations=ITERS,
            rewire_radius=0.3, w_time=0.2, w_dist=0.5)
        try: path=p.plan(goal_bias=0.2)
        except Exception: pass
    if path:
        err=float(np.linalg.norm(np.asarray(path[-1])-np.asarray(GOAL)))
        ok.append((len(path),err))
dt=(time.time()-t0)/N
if ok:
    print(f"{len(ok)}/{N} success | {dt:.1f}s/run | waypoints {np.mean([o[0] for o in ok]):.1f} | "
          f"final goal error {np.mean([o[1] for o in ok]):.2e} rad (max {max(o[1] for o in ok):.2e})")
else:
    print(f"0/{N} success | {dt:.1f}s/run")
