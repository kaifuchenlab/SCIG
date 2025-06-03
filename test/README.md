# Tests for SCIG

This directory contains unit tests for the SCIG package functions.

## Test Files

- `test_scig_human_functions.py`: Comprehensive unit tests for all functions in `scig_human_functions.py`
- `test_scig_mouse_functions.py`: Comprehensive unit tests for all functions in `scig_mouse_functions.py`

## Running Tests

### Prerequisites

1. Install the package in development mode with test dependencies:
   ```bash
   conda activate SCIG
   pip install -e ".[dev]"
   ```

### Basic Test Execution

Run all tests:
```bash
pytest test/
```

Run tests with verbose output:
```bash
pytest test/ -v
```

Run specific test file:
```bash
pytest test/test_scig_human_functions.py
pytest test/test_scig_mouse_functions.py
```

Run specific test class:
```bash
pytest test/test_scig_human_functions.py::TestPesudobulkCigpredHuman
pytest test/test_scig_mouse_functions.py::TestPesudobulkCigpredMouse
```

Run specific test method:
```bash
pytest test/test_scig_human_functions.py::TestPesudobulkCigpredHuman::test_pesudobulk_cigpred_human_rawcount
pytest test/test_scig_mouse_functions.py::TestPesudobulkCigpredMouse::test_pesudobulk_cigpred_mouse_rawcount
```

### Test Coverage

Run tests with coverage report:
```bash
pytest test/ --cov=src --cov-report=term-missing
```

Generate HTML coverage report:
```bash
pytest test/ --cov=src --cov-report=html
```

## Test Structure

### Human Function Tests (`test_scig_human_functions.py`)

#### TestPesudobulkCigpredHuman
Tests for the `pesudobulk_cigpred_human` function:
- Raw count input processing
- TPM input processing  
- Empty input edge cases

#### TestPesudobulkCigRegPredHuman
Tests for the `pesudobulk_cig_reg_pred_human` function:
- Master transcription factor prediction
- Edge case handling with minimal data

#### TestCigPredSinglecellHuman
Tests for the `cig_pred_singlecell_human` function:
- Single cell CIG prediction
- AnnData structure validation

#### TestIntegration
Integration and validation tests:
- Type annotation verification
- Import availability checks
- Data validation functions

### Mouse Function Tests (`test_scig_mouse_functions.py`)

#### TestPesudobulkCigpredMouse
Tests for the `pesudobulk_cigpred_mouse` function:
- Raw count input processing with mouse-specific features
- TPM input processing
- Empty input edge cases

#### TestPesudobulkCigRegPredMouse
Tests for the `pesudobulk_cig_reg_pred_mouse` function:
- Master transcription factor prediction for mouse
- Edge case handling with minimal data

#### TestCigPredSinglecell
Tests for the `cig_pred_singlecell` function:
- Single cell CIG prediction for mouse
- File handling and matrix operations
- Error handling for the column naming bug in the original function

#### TestIntegrationMouse
Integration tests for mouse functions:
- Type annotation verification
- Mouse-specific feature validation
- Import availability checks

## Mocking Strategy

The tests use comprehensive mocking to handle:
- **rpy2/R dependencies**: Mocked at import time to avoid requiring a full R installation
- **External libraries**: `qnorm`, `bioinfokit.norm`, `scipy.io`, etc. are mocked with appropriate return values
- **File I/O operations**: CSV reading, matrix file operations are mocked
- **ML model operations**: Scikit-learn models are tested with realistic mock data

## Mouse vs Human Function Differences

The mouse functions use different feature sets compared to human functions:
- **Mouse features**: `mca3gini`, `mtdtau_log`, `mtdroku_specificity_log`, `Total_utr_5_len`, etc.
- **Human features**: `hpa_256_tiss_ntpmtau`, `gtex_db_tissuegini`, `Total_utr_3_len`, etc.

The tests account for these differences with species-specific fixtures.

## Coverage Results

- **Human functions**: 100% coverage (317 lines)
- **Mouse functions**: 85% coverage (322 lines, 49 missing due to column naming bug)
- **Overall**: 80% coverage across all source files

## Notes

- Tests are designed to run without requiring external data files or a full R environment
- All mocking preserves the expected data types and structures
- Edge cases and error conditions are tested appropriately
- Type hints are validated for all functions
- Tests identify and properly handle existing bugs in the original code (e.g., column naming inconsistency in mouse single cell function) 