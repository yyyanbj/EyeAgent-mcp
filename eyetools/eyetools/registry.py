"""
Tool Registry for EyeAgent

This module provides a registry system for dynamically loading and managing
tools based on configuration files.
"""

import yaml
import importlib
import logging
import sys
from typing import Dict, Any, List, Type, Optional
from pathlib import Path

from .base import ToolBase

logger = logging.getLogger(__name__)

class ToolRegistry:
    """
    Registry for managing tool instances.

    Provides functionality to load tools from configuration,
    register them, and manage their lifecycle.
    """

    def __init__(self):
        self.tools: Dict[str, ToolBase] = {}
        self.tool_classes: Dict[str, Type[ToolBase]] = {}

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def register_tool_class(self, category: str, tool_class: Type[ToolBase]):
        """
        Register a tool class for a category.

        Args:
            category: Tool category name
            tool_class: Tool class to register
        """
        self.tool_classes[category] = tool_class
        logger.info(f"Registered tool class for category: {category}")

    def create_tool_from_config(self, tool_config: Dict[str, Any]) -> ToolBase:
        """
        Create a tool instance from configuration.

        Args:
            tool_config: Configuration for a single tool

        Returns:
            Tool instance
        """
        name = tool_config['name']
        category = tool_config.get('category', 'custom')
        module_path = tool_config.get('module')
        class_name = tool_config.get('class')
        params = tool_config.get('params', {})

        # Check if it's a custom tool with specific module/class
        if module_path and class_name:
            # Import the custom module
            try:
                module = importlib.import_module(module_path)
                tool_class = getattr(module, class_name)

                # Create instance
                tool_instance = tool_class(name=name, config=params)
                logger.info(f"Created custom tool instance: {name} from {module_path}.{class_name}")
                return tool_instance

            except Exception as e:
                logger.error(f"Failed to load custom tool {name}: {e}")
                raise

        # For standard tools, use the appropriate base class
        else:
            tool_instance = self._create_standard_tool(name, category, params)
            logger.info(f"Created standard tool instance: {name} of category {category}")
            return tool_instance

    def _create_standard_tool(self, name: str, category: str, params: Dict[str, Any]) -> ToolBase:
        """
        Create a standard tool instance using base classes.

        Args:
            name: Tool name
            category: Tool category
            params: Tool parameters

        Returns:
            Tool instance
        """
        from .base import ClassificationTool, DetectionTool, SegmentationTool, GenerationTool, MultimodalTool

        category_map = {
            'classification': ClassificationTool,
            'detection': DetectionTool,
            'segmentation': SegmentationTool,
            'generation': GenerationTool,
            'multimodal': MultimodalTool
        }

        tool_class = category_map.get(category.lower())
        if not tool_class:
            raise ValueError(f"Unknown tool category: {category}")

        return tool_class(name=name, config=params)

    def load_tools_from_config(self, config_path: str):
        """
        Load and register all tools from configuration file.

        Args:
            config_path: Path to the configuration file
        """
        config = self.load_config(config_path)

        for tool_config in config.get('tools', []):
            try:
                tool = self.create_tool_from_config(tool_config)
                self.tools[tool.name] = tool
            except Exception as e:
                logger.error(f"Failed to load tool {tool_config.get('name', 'unknown')}: {e}")

    def load_tool_from_zip(self, zip_path: str, extract_path: Optional[str] = None) -> ToolBase:
        """
        Load a tool from a ZIP package.

        Args:
            zip_path: Path to the ZIP file
            extract_path: Path to extract the ZIP file (optional)

        Returns:
            Tool instance
        """
        import zipfile
        import tempfile
        import shutil

        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")

        # Extract to temporary directory if not specified
        if extract_path is None:
            extract_path = tempfile.mkdtemp(prefix="eyetools_")
        else:
            extract_path = Path(extract_path)
            extract_path.mkdir(parents=True, exist_ok=True)

        # Extract ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        # Find the tool directory (should be the only directory in the ZIP)
        tool_dirs = [d for d in Path(extract_path).iterdir() if d.is_dir()]
        if len(tool_dirs) != 1:
            raise ValueError("ZIP file should contain exactly one tool directory")

        tool_dir = tool_dirs[0]

        # Validate tool package structure
        self._validate_tool_package(tool_dir)

        # Load tool configuration
        config_file = tool_dir / "config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"config.yaml not found in {tool_dir}")

        with open(config_file, 'r') as f:
            tool_config = yaml.safe_load(f)

        # Install dependencies if specified
        self._install_tool_dependencies(tool_dir, tool_config)

        # Create tool instance
        tool = self.create_tool_from_package(tool_dir, tool_config)

        logger.info(f"Loaded tool from ZIP: {tool.name}")
        return tool

    def _validate_tool_package(self, tool_dir: Path):
        """Validate that the tool package has required structure."""
        required_files = ["__init__.py", "main.py", "config.yaml", "requirements.txt"]

        for file in required_files:
            if not (tool_dir / file).exists():
                raise FileNotFoundError(f"Required file missing: {file}")

    def _install_tool_dependencies(self, tool_dir: Path, tool_config: Dict[str, Any]):
        """Install tool dependencies."""
        import subprocess
        import sys

        requirements_file = tool_dir / "requirements.txt"
        if not requirements_file.exists():
            return

        # Get the Python executable for this tool
        python_cmd = self._get_tool_python_command(tool_config)

        # Install dependencies
        try:
            cmd = f"{python_cmd} -m pip install -r {requirements_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                logger.warning(f"Failed to install dependencies: {result.stderr}")
            else:
                logger.info("Dependencies installed successfully")

        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")

    def _get_tool_python_command(self, tool_config: Dict[str, Any]) -> str:
        """Get the Python command for a tool based on its environment config."""
        environment = tool_config.get('environment', {})

        if environment.get('python_executable'):
            return environment['python_executable']
        elif environment.get('venv_path'):
            venv_path = Path(environment['venv_path'])
            python_path = venv_path / 'bin' / 'python'
            if not python_path.exists():
                python_path = venv_path / 'Scripts' / 'python.exe'  # Windows
            return str(python_path)
        elif environment.get('conda_env'):
            return f"conda run -n {environment['conda_env']} python"
        else:
            return sys.executable

    def create_tool_from_package(self, tool_dir: Path, tool_config: Dict[str, Any]) -> ToolBase:
        """
        Create a tool instance from a package directory.

        Args:
            tool_dir: Path to the tool package directory
            tool_config: Tool configuration

        Returns:
            Tool instance
        """
        name = tool_config['name']
        category = tool_config.get('category', 'custom')

        # Add tool directory to Python path
        import sys
        if str(tool_dir) not in sys.path:
            sys.path.insert(0, str(tool_dir))

        # Import the tool module
        try:
            tool_module = importlib.import_module(f"{tool_dir.name}.tool")
            tool_class = getattr(tool_module, 'ToolClass')

            # Create instance
            tool_instance = tool_class(name=name, config=tool_config)
            logger.info(f"Created tool instance from package: {name}")
            return tool_instance

        except Exception as e:
            logger.error(f"Failed to load tool from package {name}: {e}")
            raise

    def get_tool(self, name: str) -> Optional[ToolBase]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(name)

    def get_tools_by_category(self, category: str) -> List[ToolBase]:
        """
        Get all tools of a specific category.

        Args:
            category: Tool category

        Returns:
            List of tool instances
        """
        return [tool for tool in self.tools.values() if tool.__class__.__name__.lower().startswith(category.lower())]

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all registered tools with their information.

        Returns:
            List of tool information dictionaries
        """
        tools_info = []
        for name, tool in self.tools.items():
            tools_info.append({
                'name': name,
                'category': tool.__class__.__name__,
                'description': tool.get_description(),
                'enabled': tool.is_enabled(),
                'input_schema': tool.get_input_schema(),
                'output_schema': tool.get_output_schema()
            })
        return tools_info

    def enable_tool(self, name: str) -> bool:
        """
        Enable a tool by name.

        Args:
            name: Tool name

        Returns:
            True if successful, False otherwise
        """
        tool = self.get_tool(name)
        if tool:
            tool.enable()
            return True
        return False

    def disable_tool(self, name: str) -> bool:
        """
        Disable a tool by name.

        Args:
            name: Tool name

        Returns:
            True if successful, False otherwise
        """
        tool = self.get_tool(name)
        if tool:
            tool.disable()
            return True
        return False
