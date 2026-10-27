# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class DobotQuadRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "dobot_quad_rough"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DobotQuadFlatPPORunnerCfg(DobotQuadRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 500
        self.experiment_name = "dobot_quad_flat"
        self.actor.hidden_dims = [128, 128, 128]
        self.critic.hidden_dims = [128, 128, 128]


@configclass
class DobotQuadWheelRoughPPORunnerCfg(DobotQuadRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "dobot_quad_wheel_rough"


@configclass
class DobotQuadWheelStairWalkPPORunnerCfg(DobotQuadRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "dobot_quad_wheel_walk"


@configclass
class DobotQuadWheelRoughPayloadPPORunnerCfg(DobotQuadRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "dobot_quad_wheel_rough_payload"


@configclass
class DobotQuadWheelStairWalkPayloadPPORunnerCfg(DobotQuadRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "dobot_quad_wheel_walk_payload"


@configclass
class DobotQuadWheelDancePPORunnerCfg(DobotQuadRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "dobot_quad_wheel_dance"


@configclass
class DobotQuadWheelRearUpPPORunnerCfg(DobotQuadRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 2000
        self.experiment_name = "dobot_quad_wheel_rearup"


@configclass
class DobotQuadWheelFlatPPORunnerCfg(DobotQuadFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "dobot_quad_wheel_flat"


def _deploy(runner_cfg: RslRlOnPolicyRunnerCfg) -> None:
    """Feeds the critic its privileged observation group and tags the experiment."""
    runner_cfg.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    runner_cfg.experiment_name += "_deploy"


@configclass
class DobotQuadDeployFlatPPORunnerCfg(DobotQuadFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        _deploy(self)


@configclass
class DobotQuadDeployRoughPPORunnerCfg(DobotQuadRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        _deploy(self)


@configclass
class DobotQuadWheelDeployFlatPPORunnerCfg(DobotQuadWheelFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        _deploy(self)


@configclass
class DobotQuadWheelDeployRoughPPORunnerCfg(DobotQuadWheelRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        _deploy(self)
