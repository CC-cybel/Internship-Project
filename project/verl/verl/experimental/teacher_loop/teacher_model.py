# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os
from copy import deepcopy

from omegaconf import DictConfig

from verl.single_controller.ray.base import RayResourcePool, split_resource_pool
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.ray_utils import auto_await
from verl.workers.config import DistillationConfig, DistillationTeacherModelConfig, HFModelConfig
from verl.workers.rollout.replica import get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _get_teacher_route_field(route, field_name: str):
    if isinstance(route, dict):
        return route.get(field_name)
    return getattr(route, field_name)


class TeacherModelManager:
    """Teacher model manager."""

    def __init__(
        self,
        config: DictConfig,
        resource_pool: RayResourcePool = None,
    ):
        """
        Initialize the teacher model manager.

        Args:
            config (DictConfig): Teacher model configuration.
            resource_pool (RayResourcePool, optional): Resource pool. Defaults to None.
        """

        # Need dataclass conversion for max_logprobs handling in post_init
        self.config: DistillationConfig = omega_conf_to_dataclass(config)
        self.resource_pool = resource_pool
        self.route_server_managers = {}
        self.route_rollout_replicas = {}
        self.route_server_addresses = {}
        self.route_load_balancer_handles = {}
        self.route_pad_token_ids = {}
        self._use_teacher_routes = bool(self.config.teacher_routes)
        self._initialize_llm_servers()
        self._initialize_async_server_manager()
        self._initialize_router()

        self.sleep()

    def _initialize_llm_servers(self):
        if self._use_teacher_routes:
            self._initialize_routed_llm_servers()
            return

        teacher_model_config: DistillationTeacherModelConfig = self.config.teacher_model
        rollout_replicas, server_handles, server_addresses, pad_token_id = self._create_rollout_replicas(
            teacher_model_config=teacher_model_config,
            model_path=teacher_model_config.model_path,
            replica_offset=0,
            num_replicas_override=None,
            split_resource_offset=0,
        )
        self.pad_token_id = pad_token_id
        self.rollout_replicas = rollout_replicas
        self.server_handles = server_handles
        self.server_addresses = server_addresses

    def _create_rollout_replicas(
        self,
        teacher_model_config: DistillationTeacherModelConfig,
        model_path: str,
        replica_offset: int,
        num_replicas_override: int | None,
        split_resource_offset: int,
    ):
        teacher_world_size = (
            teacher_model_config.inference.tensor_model_parallel_size
            * teacher_model_config.inference.data_parallel_size
            * teacher_model_config.inference.pipeline_model_parallel_size
        )
        world_size = (
            self.resource_pool.world_size
            if self.resource_pool  # colocate mode
            else teacher_model_config.n_gpus_per_node * teacher_model_config.nnodes  # standalone mode
        )
        num_replicas = world_size // teacher_world_size if num_replicas_override is None else num_replicas_override
        if num_replicas <= 0:
            raise ValueError(
                f"Not enough teacher resources for {model_path=}: {world_size=}, {teacher_world_size=}, "
                f"{num_replicas_override=}."
            )

        rollout_replica_class = get_rollout_replica_class(teacher_model_config.inference.name)
        rollout_config = teacher_model_config.inference
        model_config = HFModelConfig(path=model_path)
        self.tokenizer = model_config.get_processor()
        text_tokenizer = model_config.tokenizer
        if model_config.tokenizer is None:
            raise ValueError(f"Tokenizer is required for teacher model {model_path}")
        pad_token_id = text_tokenizer.pad_token_id
        rollout_replicas = [
            rollout_replica_class(
                replica_rank=replica_offset + replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=teacher_model_config.n_gpus_per_node,
                is_teacher_model=True,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.resource_pool:
            split_resource_pools = split_resource_pool(self.resource_pool, split_size=teacher_world_size)
            split_resource_pools = split_resource_pools[split_resource_offset : split_resource_offset + num_replicas]
            assert len(split_resource_pools) == len(rollout_replicas)
            self._run_all(
                [
                    server.init_colocated(resource_pool)
                    for server, resource_pool in zip(rollout_replicas, split_resource_pools, strict=True)
                ]
            )
        else:
            self._run_all([server.init_standalone() for server in rollout_replicas])
        server_handles = [server._server_handle for server in rollout_replicas]
        server_addresses = [server._server_address for server in rollout_replicas]
        return rollout_replicas, server_handles, server_addresses, pad_token_id

    def _initialize_routed_llm_servers(self):
        teacher_model_config: DistillationTeacherModelConfig = self.config.teacher_model
        routes = self.config.teacher_routes or []
        route_names = [_get_teacher_route_field(route, "name") for route in routes]
        if len(route_names) != len(set(route_names)):
            raise ValueError(f"Duplicate distillation teacher route names: {route_names}")
        if any(
            not _get_teacher_route_field(route, "name") or not _get_teacher_route_field(route, "model_path")
            for route in routes
        ):
            raise ValueError("Each distillation teacher route must define non-empty name and model_path.")

        teacher_world_size = (
            teacher_model_config.inference.tensor_model_parallel_size
            * teacher_model_config.inference.data_parallel_size
            * teacher_model_config.inference.pipeline_model_parallel_size
        )
        world_size = (
            self.resource_pool.world_size
            if self.resource_pool
            else teacher_model_config.n_gpus_per_node * teacher_model_config.nnodes
        )
        total_replica_slots = world_size // teacher_world_size
        if total_replica_slots < len(routes):
            raise ValueError(
                "Multi-teacher distillation needs at least one teacher replica per route, but got "
                f"{total_replica_slots=} for {len(routes)} routes. Reduce teacher TP/DP or add GPUs."
            )
        replicas_per_route = max(1, total_replica_slots // len(routes))

        self.rollout_replicas = []
        self.server_handles = []
        self.server_addresses = []
        replica_offset = 0
        split_resource_offset = 0
        for route in routes:
            route_name = _get_teacher_route_field(route, "name")
            route_model_path = _get_teacher_route_field(route, "model_path")
            route_teacher_config = deepcopy(teacher_model_config)
            rollout_replicas, server_handles, server_addresses, pad_token_id = self._create_rollout_replicas(
                teacher_model_config=route_teacher_config,
                model_path=route_model_path,
                replica_offset=replica_offset,
                num_replicas_override=replicas_per_route,
                split_resource_offset=split_resource_offset,
            )
            self.route_rollout_replicas[route_name] = rollout_replicas
            self.route_server_addresses[route_name] = server_addresses
            self.route_pad_token_ids[route_name] = pad_token_id
            self.rollout_replicas.extend(rollout_replicas)
            self.server_handles.extend(server_handles)
            self.server_addresses.extend(server_addresses)
            replica_offset += len(rollout_replicas)
            split_resource_offset += len(rollout_replicas)

        first_pad_token_id = next(iter(self.route_pad_token_ids.values()))
        if any(pad_token_id != first_pad_token_id for pad_token_id in self.route_pad_token_ids.values()):
            raise ValueError(f"Teacher routes have inconsistent pad_token_id: {self.route_pad_token_ids}")
        self.pad_token_id = first_pad_token_id

    def _initialize_async_server_manager(self):
        from verl.experimental.agent_loop.agent_loop import GlobalRequestLoadBalancer
        from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager

        if self._use_teacher_routes:
            for route_name in self.route_server_addresses:
                route_addresses = self.route_server_addresses[route_name]
                route_handles = [server._server_handle for server in self.route_rollout_replicas[route_name]]
                load_balancer_handle = GlobalRequestLoadBalancer.remote(
                    server_actor_ids=route_addresses,
                )
                self.route_load_balancer_handles[route_name] = load_balancer_handle
                self.route_server_managers[route_name] = AsyncTeacherLLMServerManager(
                    config=self.config,
                    servers=list(zip(route_addresses, route_handles, strict=True)),
                    load_balancer_handle=load_balancer_handle,
                    distillation_config=self.config,
                    pad_token_id=self.route_pad_token_ids[route_name],
                    route_name=route_name,
                )
            self.load_balancer_handle = None
            self.server_manager = None
            return

        self.load_balancer_handle = GlobalRequestLoadBalancer.remote(
            server_actor_ids=self.server_addresses,
        )
        self.server_manager = AsyncTeacherLLMServerManager(
            config=self.config,
            servers=list(zip(self.server_addresses, self.server_handles, strict=True)),
            load_balancer_handle=self.load_balancer_handle,
            distillation_config=self.config,
            pad_token_id=self.pad_token_id,
        )

    def _initialize_router(self):
        if self._use_teacher_routes:
            self.router_address = None
            return

        worker_urls = [f"http://{server_address}" for server_address in self.server_addresses]

        from ..reward_loop.router.naive_router import launch_router_process

        self.router_address, _ = launch_router_process(worker_urls=worker_urls)

    def get_router_address(self):
        return self.router_address

    def compute_logprobs(self, data):
        self.wake_up()
        try:
            if self._use_teacher_routes:
                teacher_route = data.non_tensor_batch.get("teacher_route")
                if teacher_route is None:
                    raise ValueError("Multi-teacher distillation requires non_tensor_batch['teacher_route'].")
                routed_manager = next(iter(self.route_server_managers.values()))
                return self._run_single(
                    routed_manager.compute_teacher_logprobs_batch_by_route(
                        data=data,
                        route_to_manager=self.route_server_managers,
                    )
                )
            return self._run_single(self.server_manager.compute_teacher_logprobs_batch(data))
        finally:
            self.sleep()

    @auto_await
    async def wake_up(self):
        """Wake up all rollout replica instances."""
        await self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    @auto_await
    async def sleep(self):
        """Sleep all rollout replica instances."""
        await self._run_all([replica.sleep() for replica in self.rollout_replicas])

    @auto_await
    async def _run_all(self, tasks: list[asyncio.Task]):
        await asyncio.gather(*tasks)

    def _run_single(self, task):
        async def run():
            return await task

        return asyncio.run(run())
