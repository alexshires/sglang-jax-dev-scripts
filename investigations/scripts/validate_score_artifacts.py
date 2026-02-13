#!/usr/bin/env python3
"""
Validate Score API Benchmark Artifacts

Validates JSON artifact files against expected schemas for:
- Canonical workload definitions
- Matrix benchmark results
- Cross-backend comparison reports

Usage:
    python validate_score_artifacts.py --workload workload.json
    python validate_score_artifacts.py --matrix matrix_results.json
    python validate_score_artifacts.py --comparison comparison.json
    python validate_score_artifacts.py --all artifacts/

Exit codes:
    0 - All validations passed
    1 - Validation errors found
    2 - Invalid arguments or file not found
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    file_path: str
    schema_type: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class SchemaValidator:
    """Validates JSON artifacts against expected schemas."""

    WORKLOAD_REQUIRED_FIELDS = {
        "schema_version": str,
        "model": str,
        "query_tokens": int,
        "num_items": int,
        "item_tokens": int,
    }

    WORKLOAD_OPTIONAL_FIELDS = {
        "label_token_ids": list,
        "requests": list,
    }

    MATRIX_REQUIRED_FIELDS = {
        "schema_version": str,
        "backend": str,
        "hardware": str,
        "results": list,
    }

    MATRIX_OPTIONAL_FIELDS = {
        "server_config": dict,
        "workload_ref": str,
        "summary": dict,
    }

    MATRIX_RESULT_REQUIRED_FIELDS = {
        "chunk_size": int,
    }

    MATRIX_RESULT_OPTIONAL_FIELDS = {
        "throughput_items_per_sec": (int, float),
        "latency_p50_ms": (int, float),
        "latency_p99_ms": (int, float),
        "memory_peak_mb": (int, float),
        "status": str,
        "error": str,
    }

    COMPARISON_REQUIRED_FIELDS = {
        "schema_version": str,
    }

    COMPARISON_OPTIONAL_FIELDS = {
        "portable_view": dict,
        "best_native_view": dict,
        "correctness": dict,
        "recommendations": str,
        "notes": str,
    }

    def validate_workload(self, data: Dict[str, Any], file_path: str) -> ValidationResult:
        """Validate canonical workload JSON schema."""
        errors = []
        warnings = []

        # Check required fields
        for field, expected_type in self.WORKLOAD_REQUIRED_FIELDS.items():
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(data[field], expected_type):
                errors.append(f"Field '{field}' should be {expected_type.__name__}, got {type(data[field]).__name__}")

        # Check optional fields if present
        for field, expected_type in self.WORKLOAD_OPTIONAL_FIELDS.items():
            if field in data and not isinstance(data[field], expected_type):
                warnings.append(f"Field '{field}' should be {expected_type.__name__}, got {type(data[field]).__name__}")

        # Validate workload contract values
        if data.get("query_tokens") and data["query_tokens"] != 2000:
            warnings.append(f"query_tokens={data['query_tokens']} differs from contract (2000)")
        if data.get("num_items") and data["num_items"] != 500:
            warnings.append(f"num_items={data['num_items']} differs from contract (500)")
        if data.get("item_tokens") and data["item_tokens"] != 20:
            warnings.append(f"item_tokens={data['item_tokens']} differs from contract (20)")

        # Validate requests structure if present
        if "requests" in data and isinstance(data["requests"], list):
            for i, req in enumerate(data["requests"]):
                if not isinstance(req, dict):
                    errors.append(f"requests[{i}] should be a dict")
                elif "query" not in req:
                    warnings.append(f"requests[{i}] missing 'query' field")

        return ValidationResult(
            file_path=file_path,
            schema_type="workload",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_matrix(self, data: Dict[str, Any], file_path: str) -> ValidationResult:
        """Validate matrix benchmark results JSON schema."""
        errors = []
        warnings = []

        # Check required fields
        for field, expected_type in self.MATRIX_REQUIRED_FIELDS.items():
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(data[field], expected_type):
                errors.append(f"Field '{field}' should be {expected_type.__name__}, got {type(data[field]).__name__}")

        # Check optional fields if present
        for field, expected_type in self.MATRIX_OPTIONAL_FIELDS.items():
            if field in data and not isinstance(data[field], expected_type):
                warnings.append(f"Field '{field}' should be {expected_type.__name__}, got {type(data[field]).__name__}")

        # Validate backend value
        if data.get("backend") and data["backend"] not in ("jax", "pytorch"):
            warnings.append(f"Unexpected backend: {data['backend']} (expected 'jax' or 'pytorch')")

        # Validate results array
        if "results" in data and isinstance(data["results"], list):
            for i, result in enumerate(data["results"]):
                if not isinstance(result, dict):
                    errors.append(f"results[{i}] should be a dict")
                    continue

                # Check result required fields
                for field, expected_type in self.MATRIX_RESULT_REQUIRED_FIELDS.items():
                    if field not in result:
                        errors.append(f"results[{i}] missing required field: {field}")
                    elif not isinstance(result[field], expected_type):
                        errors.append(f"results[{i}].{field} should be {expected_type.__name__}")

                # Check result optional fields
                for field, expected_types in self.MATRIX_RESULT_OPTIONAL_FIELDS.items():
                    if field in result:
                        if isinstance(expected_types, tuple):
                            if not isinstance(result[field], expected_types):
                                warnings.append(f"results[{i}].{field} has unexpected type")
                        elif not isinstance(result[field], expected_types):
                            warnings.append(f"results[{i}].{field} has unexpected type")

        # Validate summary if present
        if "summary" in data and isinstance(data["summary"], dict):
            summary = data["summary"]
            if "best_chunk_size" in summary and summary["best_chunk_size"] is not None:
                if not isinstance(summary["best_chunk_size"], int):
                    warnings.append("summary.best_chunk_size should be an int")
            if "best_throughput" in summary and summary["best_throughput"] is not None:
                if not isinstance(summary["best_throughput"], (int, float)):
                    warnings.append("summary.best_throughput should be a number")

        # Check dense mode enforcement
        if "server_config" in data and isinstance(data["server_config"], dict):
            config = data["server_config"]
            if config.get("multi_item_mask_impl") != "dense":
                warnings.append(f"server_config.multi_item_mask_impl={config.get('multi_item_mask_impl')} (expected 'dense' for TPU)")
            if config.get("multi_item_segment_fallback_threshold", 0) != 0:
                warnings.append("server_config.multi_item_segment_fallback_threshold should be 0 for dense mode")

        return ValidationResult(
            file_path=file_path,
            schema_type="matrix",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_comparison(self, data: Dict[str, Any], file_path: str) -> ValidationResult:
        """Validate cross-backend comparison JSON schema."""
        errors = []
        warnings = []

        # Check required fields
        for field, expected_type in self.COMPARISON_REQUIRED_FIELDS.items():
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(data[field], expected_type):
                errors.append(f"Field '{field}' should be {expected_type.__name__}, got {type(data[field]).__name__}")

        # Check optional fields if present
        for field, expected_type in self.COMPARISON_OPTIONAL_FIELDS.items():
            if field in data and not isinstance(data[field], expected_type):
                warnings.append(f"Field '{field}' should be {expected_type.__name__}, got {type(data[field]).__name__}")

        # Validate portable_view if present
        if "portable_view" in data and isinstance(data["portable_view"], dict):
            pv = data["portable_view"]
            for key in ("jax_throughput", "pytorch_throughput", "ratio"):
                if key in pv and not isinstance(pv[key], (int, float)):
                    warnings.append(f"portable_view.{key} should be a number")

        # Validate correctness if present
        if "correctness" in data and isinstance(data["correctness"], dict):
            corr = data["correctness"]
            if "pass" in corr and not isinstance(corr["pass"], bool):
                warnings.append("correctness.pass should be a boolean")
            if "max_score_diff" in corr and not isinstance(corr["max_score_diff"], (int, float)):
                warnings.append("correctness.max_score_diff should be a number")

        return ValidationResult(
            file_path=file_path,
            schema_type="comparison",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


def load_json(file_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, f"File not found: {file_path}"
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"


def print_result(result: ValidationResult, verbose: bool = False) -> None:
    """Print validation result."""
    status = "PASS" if result.is_valid else "FAIL"
    print(f"[{status}] {result.file_path} ({result.schema_type})")

    if result.errors:
        for error in result.errors:
            print(f"  ERROR: {error}")

    if verbose and result.warnings:
        for warning in result.warnings:
            print(f"  WARNING: {warning}")


def validate_directory(
    dir_path: str,
    validator: SchemaValidator,
    verbose: bool = False
) -> List[ValidationResult]:
    """Validate all JSON files in a directory."""
    results = []
    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        print(f"ERROR: Not a directory: {dir_path}", file=sys.stderr)
        return results

    for json_file in dir_path.glob("**/*.json"):
        data, error = load_json(str(json_file))
        if error:
            results.append(ValidationResult(
                file_path=str(json_file),
                schema_type="unknown",
                is_valid=False,
                errors=[error],
                warnings=[],
            ))
            continue

        # Auto-detect schema type
        if "requests" in data or ("query_tokens" in data and "num_items" in data):
            result = validator.validate_workload(data, str(json_file))
        elif "results" in data and "backend" in data:
            result = validator.validate_matrix(data, str(json_file))
        elif "portable_view" in data or "best_native_view" in data:
            result = validator.validate_comparison(data, str(json_file))
        elif "server_config" in data and "results" in data:
            result = validator.validate_matrix(data, str(json_file))
        else:
            # Try to validate as matrix (most common)
            result = validator.validate_matrix(data, str(json_file))
            if not result.is_valid:
                result.warnings.append("Could not auto-detect schema type")

        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate Score API benchmark artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--workload", type=str, help="Validate a canonical workload JSON file")
    parser.add_argument("--matrix", type=str, help="Validate a matrix results JSON file")
    parser.add_argument("--comparison", type=str, help="Validate a comparison JSON file")
    parser.add_argument("--all", type=str, metavar="DIR", help="Validate all JSON files in directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show warnings in addition to errors")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    if not any([args.workload, args.matrix, args.comparison, args.all]):
        parser.print_help()
        sys.exit(2)

    validator = SchemaValidator()
    results: List[ValidationResult] = []

    # Validate specific files
    if args.workload:
        data, error = load_json(args.workload)
        if error:
            print(f"ERROR: {error}", file=sys.stderr)
            sys.exit(2)
        results.append(validator.validate_workload(data, args.workload))

    if args.matrix:
        data, error = load_json(args.matrix)
        if error:
            print(f"ERROR: {error}", file=sys.stderr)
            sys.exit(2)
        results.append(validator.validate_matrix(data, args.matrix))

    if args.comparison:
        data, error = load_json(args.comparison)
        if error:
            print(f"ERROR: {error}", file=sys.stderr)
            sys.exit(2)
        results.append(validator.validate_comparison(data, args.comparison))

    # Validate directory
    if args.all:
        results.extend(validate_directory(args.all, validator, args.verbose))

    # Output results
    if args.json:
        output = {
            "total": len(results),
            "passed": sum(1 for r in results if r.is_valid),
            "failed": sum(1 for r in results if not r.is_valid),
            "results": [
                {
                    "file": r.file_path,
                    "schema": r.schema_type,
                    "valid": r.is_valid,
                    "errors": r.errors,
                    "warnings": r.warnings,
                }
                for r in results
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        for result in results:
            print_result(result, args.verbose)

        print()
        passed = sum(1 for r in results if r.is_valid)
        failed = sum(1 for r in results if not r.is_valid)
        print(f"Summary: {passed} passed, {failed} failed, {len(results)} total")

    # Exit code
    if any(not r.is_valid for r in results):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
