import pytest
import yaml

def test_docker_compose_config():
    with open("docker-compose.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    assert "services" in config
    
    # Check required services
    required_services = ["api", "app", "train", "mlflow", "prometheus", "grafana"]
    for service in required_services:
        assert service in config["services"]
    
    # Check API service configuration
    api_service = config["services"]["api"]
    assert "build" in api_service
    assert "ports" in api_service
    assert "volumes" in api_service
    
    # Check volume configuration
    assert "volumes" in config
    assert "models" in config["volumes"]