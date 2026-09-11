import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def export_run(run_dir: Path, tb_root: Path):
    run_name = run_dir.name
    # Identify algorithm name from parent or directory name
    algo_slug = run_dir.parent.parent.name  # e.g., pnn or matd3
    tb_run_dir = tb_root / f"{algo_slug}_{run_name}"
    
    train_file = run_dir / "training_history.json"
    val_file = run_dir / "validation_history.json"
    
    if not train_file.exists():
        return False
        
    print(f"Exporting {run_name} to {tb_run_dir}...")
    writer = SummaryWriter(log_dir=str(tb_run_dir))
    
    # Export training history
    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)
        
    for ep, record in enumerate(train_data, start=1):
        if "team_reward" in record:
            writer.add_scalar("Train/Team_Reward", record["team_reward"], ep)
        if "shaping_reward" in record:
            writer.add_scalar("Train/Shaping_Reward", record["shaping_reward"], ep)
        if "team_profit" in record:
            writer.add_scalar("Train/Team_Profit", record["team_profit"], ep)
        if "profit_by_microgrid" in record:
            for mg, val in record["profit_by_microgrid"].items():
                writer.add_scalar(f"Train_Profit/{mg.upper()}", val, ep)
                
    # Export validation history
    if val_file.exists():
        with open(val_file, "r", encoding="utf-8") as f:
            val_data = json.load(f)
            
        for record in val_data:
            ep = record.get("episode", 0)
            if "team_profit_mean" in record:
                writer.add_scalar("Validation/Team_Profit_Mean", record["team_profit_mean"], ep)
            if "renewable_utilization_rate" in record:
                writer.add_scalar("Validation/Renewable_Utilization", record["renewable_utilization_rate"], ep)
            if "p2p_total_energy_kwh" in record:
                writer.add_scalar("Validation/P2P_Energy_kWh", record["p2p_total_energy_kwh"], ep)
            if "feasible_rate" in record:
                writer.add_scalar("Validation/Feasible_Rate", record["feasible_rate"], ep)
            if "profit_by_microgrid_mean" in record:
                for mg, val in record["profit_by_microgrid_mean"].items():
                    writer.add_scalar(f"Validation_Profit/{mg.upper()}", val, ep)
                    
    writer.flush()
    writer.close()
    return True

def main():
    root = Path("outputs")
    tb_root = root / "tensorboard"
    tb_root.mkdir(parents=True, exist_ok=True)
    
    # Target our primary comparison runs
    target_runs = [
        Path("outputs/pnn/runs/PNN_2026-09-05_18-11-21_seed42"),
        Path("outputs/matd3/runs/MATD3_2026-09-03_16-01-59_seed42"),
    ]
    
    for r in target_runs:
        if r.exists():
            export_run(r, tb_root)
            
    print(f"\nExport complete! TensorBoard logs written to: {tb_root}")
    print(f"To launch TensorBoard, run:")
    print(f"  tensorboard --logdir {tb_root}")

if __name__ == "__main__":
    main()
