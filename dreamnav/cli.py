import argparse
import json
import os
import sys
import time
from pathlib import Path

from .config import EnvConfig, MPCConfig, layout_seed, save_json


def environment_args(parser):
    parser.add_argument("--scenario", choices=["empty", "sparse", "dense"], default="sparse")
    parser.add_argument("--waypoints", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=600)


def planner_args(parser):
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=4)


def main():
    try:
        return _main()
    except KeyboardInterrupt:
        print("\nDreamNav stopped (Ctrl+C).", file=sys.stderr, flush=True)
        return 130


def _main():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/dreamnav-matplotlib")
    parser = argparse.ArgumentParser(description="DreamNav: simulation, world models, MPC and PPO")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="Generate complete flight episodes")
    environment_args(collect)
    collect.add_argument("--data", default="data/flights")
    collect.add_argument("--split", choices=["train", "val", "test"], default="train")
    collect.add_argument("--transitions", type=int, default=200_000)
    collect.add_argument("--seed", type=int, default=0)
    train = sub.add_parser("train-world", help="Train a GRU dynamics model")
    train.add_argument("--data", default="data/flights")
    train.add_argument("--output", default="runs/world")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--sequence-length", type=int, default=32)
    train.add_argument("--rollout-steps", type=int, default=5)
    train.add_argument("--device", default="cpu")
    train.add_argument("--seed", type=int, default=0)
    ppo = sub.add_parser("train-ppo", help="Train the matched PPO baseline")
    environment_args(ppo)
    ppo.add_argument("--output", default="runs/ppo")
    ppo.add_argument("--transitions", type=int, default=200_000)
    ppo.add_argument("--envs", type=int, default=4)
    ppo.add_argument("--seed", type=int, default=0)
    forecasts = sub.add_parser("eval-world", help="Evaluate 1/5/10/20-step prediction errors")
    forecasts.add_argument("--world", required=True)
    forecasts.add_argument("--data", default="data/flights")
    forecasts.add_argument("--output", default="runs/forecasts")
    forecasts.add_argument("--split", choices=["val", "test"], default="test")
    forecasts.add_argument("--max-windows", type=int, default=256)
    evaluation = sub.add_parser("evaluate", help="Compare controllers on identical held-out courses")
    environment_args(evaluation)
    planner_args(evaluation)
    evaluation.add_argument("--controllers", nargs="+", choices=["guided", "hover", "mpc", "ppo"], default=["mpc", "ppo"])
    evaluation.add_argument("--world")
    evaluation.add_argument("--ppo")
    evaluation.add_argument("--output", default="runs/evaluation")
    evaluation.add_argument("--episodes", type=int, default=30)
    evaluation.add_argument("--seed", type=int, default=0)
    evaluation.add_argument("--split", choices=["val", "test"], default="test")
    demo = sub.add_parser("demo", help="Run a visible flight or export an annotated video")
    environment_args(demo)
    planner_args(demo)
    demo.add_argument("--controller", choices=["guided", "hover", "mpc", "ppo"], default="guided")
    demo.add_argument("--world")
    demo.add_argument("--ppo")
    demo.add_argument("--gui", action="store_true")
    demo.add_argument("--video", help="Output .mp4 file; no GUI required")
    demo.add_argument("--output", default="runs/demo")
    demo.add_argument("--seed", type=int, default=0)
    smoke = sub.add_parser("smoke", help="Small end-to-end integration run, not a performance benchmark")
    smoke.add_argument("--output", default="runs/smoke")
    args = parser.parse_args()
    started = time.perf_counter()
    print("Loading PyTorch... Please wait; first startup can take longer.", file=sys.stderr, flush=True)
    import torch
    torch.set_num_threads(1)
    print(f"PyTorch ready ({time.perf_counter() - started:.1f}s). Loading DreamNav...", file=sys.stderr, flush=True)
    try:
        if hasattr(args, "scenario"):
            config = EnvConfig(scenario=args.scenario, waypoints=args.waypoints, max_steps=args.max_steps)
        if hasattr(args, "horizon"):
            mpc = MPCConfig(horizon=args.horizon, candidates=args.candidates, elites=args.elites, iterations=args.iterations)
        if args.command == "collect":
            from .data import collect
            result = collect(args.data, config, args.split, args.transitions, args.seed)
        elif args.command == "train-world":
            from .train import train_world
            result = train_world(args.data, args.output, args.epochs, args.batch_size, args.sequence_length,
                                 args.seed, args.device, args.rollout_steps)
        elif args.command == "train-ppo":
            from .train import train_ppo
            result = train_ppo(config, args.output, args.transitions, args.seed, args.envs)
        elif args.command == "eval-world":
            from .evaluate import evaluate_world
            result = evaluate_world(args.world, args.data, args.output, args.split, args.max_windows)
        elif args.command == "evaluate":
            from .evaluate import evaluate
            result = evaluate(config, args.output, args.controllers, args.world, args.ppo, mpc, args.episodes, args.seed, args.split)
        elif args.command == "demo":
            import pybullet as p
            from .env import DreamNavEnv
            from .evaluate import Controller, run_episode
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            if args.video:
                Path(args.video).parent.mkdir(parents=True, exist_ok=True)
            controller = Controller(args.controller, config, args.world if args.controller == "mpc" else args.ppo, mpc, args.seed)
            env = DreamNavEnv(config, render_mode="human" if args.gui else "rgb_array")
            try:
                print("Opening simulator window..." if args.gui else "Running simulation...", file=sys.stderr, flush=True)
                result, path = run_episode(env, controller, layout_seed("test", args.seed), args.video, args.gui)
                from PIL import Image
                Image.fromarray(env.render()).save(output / "final_frame.png")
                import numpy as np
                np.save(output / "trajectory.npy", path)
                save_json(output / "result.json", result)
            except p.error:
                if args.gui and env.aviary is not None and not p.isConnected(env.client):
                    print("Simulator disconnected; demo stopped.", file=sys.stderr, flush=True)
                    return 0
                raise
            finally:
                env.close()
                controller.close()
        else:
            result = smoke_run(args.output)
        print(json.dumps(result, indent=2), flush=True)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


def smoke_run(output):
    from .data import collect
    from .train import train_world, train_ppo
    from .evaluate import evaluate, evaluate_world
    output = Path(output)
    cfg = EnvConfig(max_steps=100)
    for split, count in [("train", 1200), ("val", 300), ("test", 300)]:
        collect(output / "data", cfg, split, count)
    train_world(output / "data", output / "world", epochs=3, batch_size=16)
    train_ppo(cfg, output / "ppo", transitions=512, n_envs=2)
    evaluate_world(output / "world/world.pt", output / "data", output / "forecasts", max_windows=32)
    result = evaluate(cfg, output / "evaluation", ["guided", "mpc", "ppo"], output / "world/world.pt",
                      output / "ppo/ppo.zip", MPCConfig(horizon=5, candidates=32, elites=8, iterations=2), episodes=2)
    result["purpose"] = "Pipeline integration only. These tiny training budgets do not establish navigation quality."
    save_json(output / "smoke.json", result)
    return result
