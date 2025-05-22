import pytest
import yaml
import json
import os

def test_prometheus_config():
    with open("monitoring/prometheus.yml", "r") as file:
        config = yaml.safe_load(file)
    
    assert "global" in config
    assert "scrape_configs" in config
    assert isinstance(config["scrape_configs"], list)
    
    # Check for API job
    api_job = next((job for job in config["scrape_configs"] if job["job_name"] == "api"), None)
    assert api_job is not None
    assert "dns_sd_configs" in api_job

def test_datasource_config():
    with open("monitoring/datasource.yml", "r") as file:
        config = yaml.safe_load(file)
    
    assert "apiVersion" in config
    assert "datasources" in config
    assert isinstance(config["datasources"], list)
    
    # Check Prometheus datasource
    prometheus_ds = next((ds for ds in config["datasources"] if ds["name"] == "Prometheus"), None)
    assert prometheus_ds is not None
    assert prometheus_ds["type"] == "prometheus"
    assert prometheus_ds["url"] == "http://prometheus:9090"

def test_dashboard_json():
    with open("monitoring/dashboard.json", "r") as file:
        dashboard = json.load(file)
    
    assert "panels" in dashboard
    assert isinstance(dashboard["panels"], list)
    assert len(dashboard["panels"]) > 0
    
    # Check for model metrics panels
    model_prediction_panel = next((panel for panel in dashboard["panels"] 
                                  if panel.get("title") == "Predictions per minute by result"), None)
    assert model_prediction_panel is not None