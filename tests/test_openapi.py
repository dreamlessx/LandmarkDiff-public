"""Tests for OpenAPI specification validity."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

SPEC_PATH = Path(__file__).parent.parent / "docs" / "api" / "openapi.yaml"


@pytest.fixture
def spec():
    return yaml.safe_load(SPEC_PATH.read_text())


class TestOpenAPISpec:
    def test_file_exists(self):
        assert SPEC_PATH.exists()

    def test_valid_yaml(self):
        data = yaml.safe_load(SPEC_PATH.read_text())
        assert isinstance(data, dict)

    def test_openapi_version(self, spec):
        assert spec["openapi"].startswith("3.0")

    def test_info_present(self, spec):
        assert "info" in spec
        assert "title" in spec["info"]
        assert "version" in spec["info"]

    def test_paths_present(self, spec):
        assert "paths" in spec
        assert len(spec["paths"]) >= 3

    def test_health_endpoint(self, spec):
        assert "/health" in spec["paths"]
        assert "get" in spec["paths"]["/health"]

    def test_predict_endpoint(self, spec):
        assert "/predict" in spec["paths"]
        predict = spec["paths"]["/predict"]
        assert "post" in predict
        assert "requestBody" in predict["post"]

    def test_analyze_endpoint(self, spec):
        assert "/analyze" in spec["paths"]
        assert "post" in spec["paths"]["/analyze"]

    def test_procedures_endpoint(self, spec):
        assert "/procedures" in spec["paths"]
        assert "get" in spec["paths"]["/procedures"]

    def test_all_17_procedures_listed(self, spec):
        predict = spec["paths"]["/predict"]["post"]
        props = predict["requestBody"]["content"]["multipart/form-data"]["schema"]["properties"]
        procedures = props["procedure"]["enum"]
        assert len(procedures) == 17
        assert "rhinoplasty" in procedures
        assert "otoplasty" in procedures

    def test_intensity_range(self, spec):
        predict = spec["paths"]["/predict"]["post"]
        props = predict["requestBody"]["content"]["multipart/form-data"]["schema"]["properties"]
        intensity = props["intensity"]
        assert intensity["minimum"] == 0
        assert intensity["maximum"] == 100

    def test_schemas_present(self, spec):
        assert "components" in spec
        assert "schemas" in spec["components"]
        schemas = spec["components"]["schemas"]
        assert "PredictionResponse" in schemas
        assert "ErrorResponse" in schemas
        assert "HealthResponse" in schemas

    def test_error_response_has_detail(self, spec):
        error = spec["components"]["schemas"]["ErrorResponse"]
        assert "detail" in error["properties"]

    def test_prediction_response_has_output_image(self, spec):
        pred = spec["components"]["schemas"]["PredictionResponse"]
        assert "output_image" in pred["properties"]

    def test_batch_endpoint(self, spec):
        assert "/batch" in spec["paths"]
        batch = spec["paths"]["/batch"]
        assert "post" in batch
