# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for swebench_imports module."""

import pytest
from prompt_siren.datasets.swebench_dataset.constants import InjectionSpec
from prompt_siren.datasets.swebench_dataset.swebench_imports import (
    _convert_conda_version_spec,
    convert_environment_yml_to_requirements,
    make_repo_script_list_py,
)


class TestConvertCondaVersionSpec:
    """Tests for _convert_conda_version_spec function."""

    def test_convert_exact_version(self):
        """Test conversion of conda exact version (=) to pip exact version (==)."""
        assert _convert_conda_version_spec("numpy=1.19") == "numpy==1.19"
        assert _convert_conda_version_spec("python=3.8") == "python==3.8"

    def test_preserve_inequality_operators(self):
        """Test that inequality operators are preserved."""
        assert _convert_conda_version_spec("numpy>=1.19") == "numpy>=1.19"
        assert _convert_conda_version_spec("pillow<=8.0") == "pillow<=8.0"
        assert _convert_conda_version_spec("pytest!=4.6.0") == "pytest!=4.6.0"
        assert _convert_conda_version_spec("setuptools~=60.0") == "setuptools~=60.0"

    def test_preserve_double_equals(self):
        """Test that double equals is preserved."""
        assert _convert_conda_version_spec("numpy==1.19") == "numpy==1.19"

    def test_no_version_spec(self):
        """Test packages without version specifications."""
        assert _convert_conda_version_spec("numpy") == "numpy"
        assert _convert_conda_version_spec("pytest") == "pytest"


class TestConvertEnvironmentYmlToRequirements:
    """Tests for convert_environment_yml_to_requirements function."""

    def test_basic_conversion(self):
        """Test basic conversion with simple packages."""
        yml = """
name: testbed
channels:
  - conda-forge
dependencies:
  - numpy>=1.19
  - pillow>=6.2
  - pytest
"""
        result = convert_environment_yml_to_requirements(yml)
        lines = result.split("\n")

        assert "numpy>=1.19" in lines
        assert "pillow>=6.2" in lines
        assert "pytest" in lines

    def test_conversion_with_pip_subsection(self):
        """Test conversion with pip subsection."""
        yml = """
name: testbed
channels:
  - conda-forge
dependencies:
  - numpy>=1.19
  - pillow>=6.2
  - pip
  - pip:
      - mpl-sphinx-theme
      - sphinxcontrib-svg2pdfconverter
"""
        result = convert_environment_yml_to_requirements(yml)
        lines = result.split("\n")

        # Conda packages should be included
        assert "numpy>=1.19" in lines
        assert "pillow>=6.2" in lines

        # Pip packages from pip: subsection should be included
        assert "mpl-sphinx-theme" in lines
        assert "sphinxcontrib-svg2pdfconverter" in lines

        # 'pip' itself should not be included
        assert "pip" not in lines

    def test_skip_python_package(self):
        """Test that python package is skipped."""
        yml = """
name: testbed
dependencies:
  - python=3.8
  - numpy>=1.19
"""
        result = convert_environment_yml_to_requirements(yml)
        lines = result.split("\n")

        # Python should be skipped
        assert not any("python" in line.lower() for line in lines)

        # Other packages should be included
        assert "numpy>=1.19" in lines

    def test_convert_exact_version_syntax(self):
        """Test that conda exact version (=) is converted to pip (==)."""
        yml = """
name: testbed
dependencies:
  - numpy=1.19.5
  - pytest=6.2.0
"""
        result = convert_environment_yml_to_requirements(yml)
        lines = result.split("\n")

        assert "numpy==1.19.5" in lines
        assert "pytest==6.2.0" in lines

    def test_matplotlib_example(self):
        """Test with real-world matplotlib environment.yml example."""
        yml = """
name: testbed
channels:
  - conda-forge
dependencies:
  - cairocffi
  - contourpy>=1.0.1
  - cycler>=0.10.0
  - fonttools>=4.22.0
  - kiwisolver>=1.0.1
  - numpy>=1.19
  - pillow>=6.2
  - pygobject
  - pyparsing
  - pyqt
  - python-dateutil>=2.1
  - setuptools
  - setuptools_scm
  - wxpython
  - pip
  - pip:
      - mpl-sphinx-theme
      - sphinxcontrib-svg2pdfconverter
      - pikepdf
"""
        result = convert_environment_yml_to_requirements(yml)
        lines = result.split("\n")

        # Check some conda packages
        assert "cairocffi" in lines
        assert "contourpy>=1.0.1" in lines
        assert "numpy>=1.19" in lines

        # Check pip packages
        assert "mpl-sphinx-theme" in lines
        assert "sphinxcontrib-svg2pdfconverter" in lines
        assert "pikepdf" in lines

        # pip should not be in the list
        assert "pip" not in lines

    def test_invalid_yaml(self):
        """Test error handling for invalid YAML."""
        yml = """
this is not: [valid yaml
"""
        with pytest.raises(ValueError, match="Invalid YAML"):
            convert_environment_yml_to_requirements(yml)

    def test_missing_dependencies(self):
        """Test error handling when dependencies section is missing."""
        yml = """
name: testbed
channels:
  - conda-forge
"""
        with pytest.raises(ValueError, match="missing 'dependencies' section"):
            convert_environment_yml_to_requirements(yml)

    def test_dependencies_not_list(self):
        """Test error handling when dependencies is not a list."""
        yml = """
name: testbed
dependencies: not-a-list
"""
        with pytest.raises(ValueError, match="'dependencies' must be a list"):
            convert_environment_yml_to_requirements(yml)

    def test_yaml_not_dict(self):
        """Test error handling when YAML root is not a dictionary."""
        yml = """
- item1
- item2
"""
        with pytest.raises(ValueError, match="must be a YAML dictionary"):
            convert_environment_yml_to_requirements(yml)

    def test_empty_dependencies(self):
        """Test handling of empty dependencies list."""
        yml = """
name: testbed
dependencies: []
"""
        result = convert_environment_yml_to_requirements(yml)
        assert result == ""

    def test_whitespace_handling(self):
        """Test that whitespace is properly stripped from package names."""
        yml = """
name: testbed
dependencies:
  - numpy>=1.19
  -   pillow
"""
        result = convert_environment_yml_to_requirements(yml)
        lines = result.split("\n")

        assert "numpy>=1.19" in lines
        assert "pillow" in lines


class TestMakeRepoScriptListPyWithInjection:
    """Tests for make_repo_script_list_py function with injection support."""

    def test_injection_commands_generated(self):
        """Test that injection commands are generated when injection_spec is provided."""
        specs = {"python": "3.9"}
        injection_spec: InjectionSpec = {
            "file": "django/conf/__init__.py",
            "line": 10,
            "content": "<injection_vector_ea2cbaa4>",
        }

        result = make_repo_script_list_py(
            specs=specs,
            repo="django/django",
            repo_directory="/testbed",
            base_commit="abc123",
            env_name="testbed",
            injection_spec=injection_spec,
        )

        # Convert to single string for easier searching
        script = "\n".join(result)

        # Check that sed command is present
        assert "sed -i '10i\\<injection_vector_ea2cbaa4>' django/conf/__init__.py" in script

        # Check that git add command is present
        assert "git add django/conf/__init__.py" in script

        # Check that injection commands come before git commit
        sed_index = script.index("sed -i")
        git_add_index = script.index("git add django/conf/__init__.py")
        git_commit_index = script.index("git commit")

        assert sed_index < git_add_index < git_commit_index

    def test_no_injection_without_spec(self):
        """Test that no injection commands are generated when injection_spec is None."""
        specs = {"python": "3.9"}

        result = make_repo_script_list_py(
            specs=specs,
            repo="django/django",
            repo_directory="/testbed",
            base_commit="abc123",
            env_name="testbed",
            injection_spec=None,
        )

        script = "\n".join(result)

        # Check that sed command for injection is NOT present
        assert "sed -i" not in script or "injection" not in script

    def test_special_character_escaping(self):
        """Test that special characters in content are escaped for sed."""
        specs = {"python": "3.9"}

        # Test backslash escaping
        injection_spec: InjectionSpec = {
            "file": "test.py",
            "line": 5,
            "content": "path\\to\\file",
        }
        result = make_repo_script_list_py(
            specs=specs,
            repo="django/django",
            repo_directory="/testbed",
            base_commit="abc123",
            env_name="testbed",
            injection_spec=injection_spec,
        )
        script = "\n".join(result)
        # Backslashes should be escaped
        assert "path\\\\to\\\\file" in script

        # Test forward slash escaping
        injection_spec = {
            "file": "test.py",
            "line": 5,
            "content": "path/to/file",
        }
        result = make_repo_script_list_py(
            specs=specs,
            repo="django/django",
            repo_directory="/testbed",
            base_commit="abc123",
            env_name="testbed",
            injection_spec=injection_spec,
        )
        script = "\n".join(result)
        # Forward slashes should be escaped
        assert "path\\/to\\/file" in script

        # Test ampersand escaping
        injection_spec = {
            "file": "test.py",
            "line": 5,
            "content": "foo & bar",
        }
        result = make_repo_script_list_py(
            specs=specs,
            repo="django/django",
            repo_directory="/testbed",
            base_commit="abc123",
            env_name="testbed",
            injection_spec=injection_spec,
        )
        script = "\n".join(result)
        # Ampersands should be escaped
        assert "foo \\& bar" in script
