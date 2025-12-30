---
name: experiment-tracking
version: "2.0.0"
sasmp_version: "1.3.0"
description: Master ML experiment tracking - MLflow, W&B, Neptune, versioning, reproducibility
bonded_agent: 02-experiment-tracking
bond_type: PRIMARY_BOND

# SKILL METADATA
category: experimentation
difficulty: intermediate
estimated_hours: 30
prerequisites:
  - mlops-basics

# VALIDATION
validation:
  pre_conditions:
    - "Completed mlops-basics skill"
    - "Python environment setup"
  post_conditions:
    - "Can set up experiment tracking"
    - "Can log parameters, metrics, artifacts"
    - "Can use model registry"
  parameter_schema:
    platform:
      type: string
      enum: [mlflow, wandb, neptune, comet]

# OBSERVABILITY
observability:
  log_inputs: true
  log_outputs: true
  metrics:
    - experiments_created
    - runs_logged
    - models_registered
---

# Experiment Tracking Skill

> **Learn**: Master ML experiment tracking for reproducibility and collaboration.

## Skill Overview

| Attribute | Value |
|-----------|-------|
| **Bonded Agent** | 02-experiment-tracking |
| **Difficulty** | Intermediate |
| **Duration** | 30 hours |
| **Prerequisites** | mlops-basics |

---

## Learning Objectives

1. **Set up** experiment tracking infrastructure
2. **Log** parameters, metrics, and artifacts systematically
3. **Compare** experiments and identify best models
4. **Use** model registry for version management
5. **Collaborate** with team using shared tracking

---

## Topics Covered

### Module 1: Platform Setup (6 hours)

**Platform Comparison:**

| Feature | MLflow | W&B | Neptune |
|---------|--------|-----|---------|
| Self-hosted | ✅ | ❌ | ❌ |
| Free tier | ✅ | ✅ | ✅ |
| Real-time | ❌ | ✅ | ✅ |
| Git integration | ⚠️ | ✅ | ✅ |

**Setup Exercises:**
- [ ] Install MLflow and start local server
- [ ] Create W&B account and initialize project
- [ ] Compare UI/UX of both platforms

---

### Module 2: Experiment Logging (10 hours)

**What to Log:**

```python
# Complete logging example
with mlflow.start_run():
    # 1. Parameters (hyperparameters, configs)
    mlflow.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "model_type": "transformer"
    })

    # 2. Metrics (per-step and final)
    for epoch in range(10):
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)

    # 3. Artifacts (models, plots, configs)
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.pytorch.log_model(model, "model")

    # 4. Tags (for filtering)
    mlflow.set_tags({
        "experiment_type": "baseline",
        "dataset_version": "v2.1"
    })
```

---

### Module 3: Model Registry (8 hours)

**Registry Workflow:**

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL REGISTRY FLOW                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Train → Log Model → Register → Staging → Production → Archive
│                          │          │           │              │
│                          ▼          ▼           ▼              │
│                     Version 1   Validate    Deploy           │
│                     Version 2   A/B Test    Monitor          │
│                     Version N   Approve     Rollback         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Exercises:**
- [ ] Register a trained model
- [ ] Promote model through stages
- [ ] Implement rollback procedure

---

### Module 4: Best Practices (6 hours)

**Naming Conventions:**
```
experiments/
├── {project_name}/
│   ├── {experiment_type}_{date}/
│   │   ├── run_{config_hash}/
```

**Reproducibility Checklist:**
- [ ] Log git commit hash
- [ ] Capture environment (pip freeze)
- [ ] Set and log random seeds
- [ ] Log data version/hash
- [ ] Save config files as artifacts

---

## Code Templates

### Template: Production Experiment Tracker

```python
# templates/experiment_tracker.py
import mlflow
import hashlib
import subprocess
from datetime import datetime

class ProductionExperimentTracker:
    """Production-ready experiment tracking wrapper."""

    def __init__(self, experiment_name: str, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = None

    def start_run(self, run_name: str = None):
        """Start a new tracked run."""
        self.run = mlflow.start_run(run_name=run_name)

        # Auto-log environment info
        self._log_environment()
        return self

    def _log_environment(self):
        """Capture reproducibility information."""
        # Git info
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            ).decode().strip()
            mlflow.set_tag("git_commit", git_hash)
        except:
            pass

        # Timestamp
        mlflow.set_tag("run_timestamp", datetime.now().isoformat())

    def log_config(self, config: dict):
        """Log configuration as parameters."""
        # Flatten nested config
        flat_config = self._flatten_dict(config)
        mlflow.log_params(flat_config)

    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics with optional step."""
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, artifact_path: str = "model"):
        """Log model with signature."""
        mlflow.pytorch.log_model(model, artifact_path)

    def end_run(self):
        """End the current run."""
        if self.run:
            mlflow.end_run()

    def _flatten_dict(self, d: dict, parent_key: str = '') -> dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
```

---

## Troubleshooting Guide

| Issue | Cause | Solution |
|-------|-------|----------|
| Runs not syncing | Network issue | Check connectivity, use offline mode |
| Large artifacts fail | Size limit | Use cloud storage for large files |
| Duplicate run names | No uniqueness | Add timestamp or hash to names |

---

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [W&B Documentation](https://docs.wandb.ai/)
- [See: training-pipelines] - Integrate tracking with pipelines

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2024-12 | Production-grade with templates |
| 1.0.0 | 2024-11 | Initial release |
