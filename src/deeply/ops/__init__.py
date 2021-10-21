from deeply.ops.integrations import (
    WeightsAndBiases
)

SERVICE_REGISTRY = {
    "wandb": WeightsAndBiases
}

def service(name):
    if name not in SERVICE_REGISTRY:
        raise ValueError("No service %s found." % name)

    service_class = SERVICE_REGISTRY[name]
    service_instance = service_class()

    return service_instance