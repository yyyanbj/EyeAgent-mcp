"""
Tool Manager for EyeAgent

This module provides functionality for managing tool packages,
including uploading, deploying, and managing tool environments.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

from .registry import ToolRegistry

logger = logging.getLogger(__name__)

class ToolManager:
    """
    Manager for tool packages and environments.

    Provides functionality to deploy tools from ZIP files,
    manage environments, and handle tool lifecycle.
    """

    def __init__(self, registry: ToolRegistry, tools_dir: str = "tools"):
        """
        Initialize the tool manager.

        Args:
            registry: Tool registry instance
            tools_dir: Directory to store deployed tools
        """
        self.registry = registry
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(exist_ok=True)

        # Track deployed tools
        self.deployed_tools: Dict[str, Dict[str, Any]] = {}

    def deploy_tool_from_zip(self, zip_path: str, tool_name: Optional[str] = None) -> str:
        """
        Deploy a tool from a ZIP package.

        Args:
            zip_path: Path to the ZIP file
            tool_name: Optional name for the tool (overrides config)

        Returns:
            Name of the deployed tool
        """
        # Load tool from ZIP
        tool = self.registry.load_tool_from_zip(zip_path, str(self.tools_dir))

        # Use provided name or from config
        final_name = tool_name or tool.name

        # Rename tool if needed
        if final_name != tool.name:
            tool.name = final_name

        # Register the tool
        self.registry.tools[final_name] = tool

        # Track deployment
        self.deployed_tools[final_name] = {
            'zip_path': zip_path,
            'extract_path': str(self.tools_dir / tool.name),
            'config': tool.config,
            'environment': tool.environment
        }

        logger.info(f"Deployed tool: {final_name}")
        return final_name

    def undeploy_tool(self, tool_name: str) -> bool:
        """
        Undeploy a tool and clean up its files.

        Args:
            tool_name: Name of the tool to undeploy

        Returns:
            True if successful, False otherwise
        """
        if tool_name not in self.deployed_tools:
            logger.warning(f"Tool not found in deployed tools: {tool_name}")
            return False

        # Remove from registry
        if tool_name in self.registry.tools:
            del self.registry.tools[tool_name]

        # Clean up files
        extract_path = self.deployed_tools[tool_name]['extract_path']
        try:
            shutil.rmtree(extract_path)
            logger.info(f"Cleaned up tool files: {extract_path}")
        except Exception as e:
            logger.error(f"Failed to clean up tool files: {e}")

        # Remove from tracking
        del self.deployed_tools[tool_name]

        logger.info(f"Undeployed tool: {tool_name}")
        return True

    def list_deployed_tools(self) -> List[Dict[str, Any]]:
        """
        List all deployed tools with their information.

        Returns:
            List of deployed tool information
        """
        tools_info = []

        for name, info in self.deployed_tools.items():
            tool = self.registry.get_tool(name)
            if tool:
                tools_info.append({
                    'name': name,
                    'description': tool.get_description(),
                    'category': info['config'].get('category', 'unknown'),
                    'version': info['config'].get('version', 'unknown'),
                    'environment': info['environment'],
                    'enabled': tool.is_enabled(),
                    'path': info['extract_path']
                })

        return tools_info

    def get_tool_environment_info(self, tool_name: str) -> Dict[str, Any]:
        """
        Get environment information for a deployed tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Environment information dictionary
        """
        if tool_name not in self.deployed_tools:
            raise ValueError(f"Tool not found: {tool_name}")

        info = self.deployed_tools[tool_name]
        tool = self.registry.get_tool(tool_name)

        return {
            'name': tool_name,
            'python_command': tool.get_execution_command() if tool else 'unknown',
            'environment_config': info['environment'],
            'dependencies': info['config'].get('environment', {}).get('dependencies', []),
            'path': info['extract_path']
        }

    def validate_tool_environment(self, tool_name: str) -> Dict[str, Any]:
        """
        Validate the environment for a deployed tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Validation results
        """
        if tool_name not in self.deployed_tools:
            return {'valid': False, 'error': 'Tool not deployed'}

        tool = self.registry.get_tool(tool_name)
        if not tool:
            return {'valid': False, 'error': 'Tool not found in registry'}

        try:
            tool.validate_environment()
            return {
                'valid': True,
                'python_command': tool.get_execution_command(),
                'message': 'Environment validation passed'
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'python_command': tool.get_execution_command()
            }

    def update_tool_config(self, tool_name: str, new_config: Dict[str, Any]) -> bool:
        """
        Update the configuration of a deployed tool.

        Args:
            tool_name: Name of the tool
            new_config: New configuration dictionary

        Returns:
            True if successful, False otherwise
        """
        if tool_name not in self.deployed_tools:
            logger.warning(f"Tool not found: {tool_name}")
            return False

        tool = self.registry.get_tool(tool_name)
        if not tool:
            logger.warning(f"Tool not found in registry: {tool_name}")
            return False

        # Update tool config
        tool.config.update(new_config)
        self.deployed_tools[tool_name]['config'].update(new_config)

        # Re-validate environment if environment config changed
        if 'environment' in new_config:
            tool.environment = new_config['environment']
            tool.python_executable = tool.environment.get('python_executable')
            tool.venv_path = tool.environment.get('venv_path')
            tool.conda_env = tool.environment.get('conda_env')

        logger.info(f"Updated configuration for tool: {tool_name}")
        return True
