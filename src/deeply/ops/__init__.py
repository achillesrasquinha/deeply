from deeply.ops.integrations import (
    WeightsAndBiases,
    AzureML
)

SERVICE_REGISTRY = {
    "wandb": WeightsAndBiases,
    "azure-ml": AzureML
}

def service(name):
    if name not in SERVICE_REGISTRY:
        raise ValueError("No service %s found." % name)

    service_class = SERVICE_REGISTRY[name]
    service_instance = service_class()

    return service_instance