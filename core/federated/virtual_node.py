from typing import Dict
from typing import Type

from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from validators.config_validator import ConfigValidator
from utils.log import Log


class VirtualNode(object):

    """
    A VirtualNode is a wrapper around a FedRayNode that is used to represent a node
    within a federation. This allows a Federation to perform the lazy initialization
    and build of the node, which is deferred to the first call of the node within
    a session.
    """

    def __init__(
        self,
        template: Type[object], # FederatedBase
        id: str,
        role: str,
        config: 'ConfigValidator',
        log: Log,
    ) -> None:
        """Creates a new VirtualNode object.

        Args:
            template (Type[FederatedBase]): The template for the node.
            id (str): The ID of the node.
            role (str): The role of the node.
            config (Dict): The configuration to be passed to the build method of the node.
        """
        self.template = template
        self.fed_id = config.FEDERATION_ID
        self.id = id
        self.role = role
        self.config = config
        self.log = log
        self.handle: object = None # FederatedBase

    def build(self, bundle_idx: int, placement_group: PlacementGroup):
        """Builds the node.

        Args:
            bundle_idx (int): The index of the bundle within the placement group.
            placement_group (PlacementGroup): The placement group to be used for the
                node.
        """
        resources = placement_group.bundle_specs[bundle_idx]
        num_cpus = resources["CPU"]
        num_gpus = resources["GPU"] if "GPU" in resources else 0
        self.handle = self.template.options(
            name="/".join([self.fed_id, self.id]),
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group, placement_group_bundle_index=bundle_idx
            ),
        ).remote(
            node_id=self.id, role=self.role, config=self.config, log=self.log
        )

    @property
    def built(self):
        """Returns whether the node has been built."""
        return self.handle is not None
