# Decorators

## @Remote

This function defines a remote node within a federated learning system.

It overrides the default behavior of the `ray.remote` decorator by enforcing a minimum max_concurrency value of 2. This adjustment mitigates the risk of a node becoming blocked by a singular active training or evaluation session. The default max_concurrency for these remote nodes is set to 100.

All other arguments accepted by this function are consistent with those of the standard `ray.remote` decorator.