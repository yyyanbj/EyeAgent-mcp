"""
Base classes for EyeAgent tools framework.

This module defines the base classes for different tool categories,
providing a unified interface for tool registration and execution.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import time
from pathlib import Path

# Import utilities
try:
    from .utils import load_image_from_path
except ImportError:
    # Fallback for when utils is not available
    load_image_from_path = None

logger = logging.getLogger(__name__)

class ToolBase(ABC):
    """
    Base class for all tools in EyeAgent.

    Provides common interface and functionality for tool registration,
    configuration, and execution.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the tool.

        Args:
            name: Unique name for the tool
            config: Configuration dictionary for the tool
        """
        self.name = name
        self.config = config or {}
        self.enabled = True
        self.device = self.config.get('device', 'cuda' if self._has_cuda() else 'cpu')
        self.model = None
        self.initialized = False

        # Environment configuration
        self.environment = self.config.get('environment', {})
        self.python_executable = self.environment.get('python_executable')
        self.venv_path = self.environment.get('venv_path')
        self.conda_env = self.environment.get('conda_env')
        self.required_python_version = self.environment.get('python_version', '>=3.8')
        self.dependencies = self.environment.get('dependencies', [])

        logger.info(f"Initialized tool: {name}")

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Tool execution result
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Get tool description for documentation.

        Returns:
            Tool description string
        """
        pass

    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for tool inputs.

        Returns:
            Input schema dictionary
        """
        pass

    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for tool outputs.

        Returns:
            Output schema dictionary
        """
        pass

    def enable(self):
        """Enable the tool."""
        self.enabled = True
        logger.info(f"Enabled tool: {self.name}")

    def disable(self):
        """Disable the tool."""
        self.enabled = False
        logger.info(f"Disabled tool: {self.name}")

    def is_enabled(self) -> bool:
        """Check if tool is enabled."""
        return self.enabled

    def validate_environment(self):
        """Validate that the required environment is available."""
        import sys
        from packaging import version

        # Check Python version
        if self.required_python_version:
            try:
                from packaging.specifiers import SpecifierSet
                spec = SpecifierSet(self.required_python_version)
                current_version = version.parse(sys.version.split()[0])
                if current_version not in spec:
                    raise EnvironmentError(f"Python version {current_version} does not meet requirement {self.required_python_version}")
            except ImportError:
                logger.warning("packaging not available for version checking")

        # Validate environment paths
        if self.python_executable:
            import os
            if not os.path.exists(self.python_executable):
                raise EnvironmentError(f"Python executable not found: {self.python_executable}")

        if self.venv_path:
            import os
            if not os.path.exists(self.venv_path):
                raise EnvironmentError(f"Virtual environment not found: {self.venv_path}")

        if self.conda_env:
            # Check if conda environment exists
            import subprocess
            try:
                result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
                if self.conda_env not in result.stdout:
                    raise EnvironmentError(f"Conda environment not found: {self.conda_env}")
            except FileNotFoundError:
                logger.warning("Conda not found, cannot validate conda environment")

        logger.info(f"Environment validation passed for tool: {self.name}")

    def get_execution_command(self) -> str:
        """Get the command to execute this tool in its environment."""
        if self.python_executable:
            return self.python_executable
        elif self.venv_path:
            import os
            python_path = os.path.join(self.venv_path, 'bin', 'python')
            if os.name == 'nt':  # Windows
                python_path = os.path.join(self.venv_path, 'Scripts', 'python.exe')
            return python_path
        elif self.conda_env:
            return f"conda run -n {self.conda_env} python"
        else:
            return "python"

    def initialize(self):
        """Initialize the tool (load model, setup transforms, etc.)"""
        if not self.initialized:
            # Validate environment first
            self.validate_environment()

            self.load_model()
            self.setup_preprocessing()
            self.initialized = True
            logger.info(f"Tool {self.name} initialized successfully")

    @abstractmethod
    def load_model(self):
        """Load the model for this tool."""
        pass

    @abstractmethod
    def setup_preprocessing(self):
        """Setup preprocessing transforms."""
        pass

    @abstractmethod
    def preprocess(self, inputs: Any) -> Any:
        """Preprocess inputs for inference."""
        pass

    @abstractmethod
    def inference(self, processed_inputs: Any) -> Any:
        """Run inference on processed inputs."""
        pass

    @abstractmethod
    def postprocess(self, raw_outputs: Any) -> Any:
        """Postprocess raw model outputs."""
        pass

    def run(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Standard execution pipeline.

        Returns:
            Tuple of (results, metadata)
        """
        start_time = time.time()

        try:
            # Ensure tool is initialized
            if not self.initialized:
                self.initialize()

            # Preprocess inputs
            processed_inputs = self.preprocess(kwargs)

            # Run inference
            raw_outputs = self.inference(processed_inputs)

            # Postprocess results
            results = self.postprocess(raw_outputs)

            # Create metadata
            metadata = {
                'tool_name': self.name,
                'execution_time': time.time() - start_time,
                'status': 'completed',
                'device': self.device
            }

            return results, metadata

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            metadata = {
                'tool_name': self.name,
                'execution_time': time.time() - start_time,
                'status': 'failed',
                'error': str(e)
            }
            return None, metadata


class ClassificationTool(ToolBase):
    """
    Base class for classification tools.

    Provides common functionality for image classification tasks.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.model_path = config.get('model_path', 'default')
        self.model_type = config.get('model_type', 'default')
        self.classes = config.get('classes', [])
        self.threshold = config.get('threshold', 0.5)
        self.task = config.get('task', 'classification')

    def execute(self, image_path: str) -> Dict[str, Any]:
        """
        Execute classification on an image.

        Args:
            image_path: Path to the image

        Returns:
            Classification result
        """
        results, metadata = self.run(image_path=image_path)
        return results

    def preprocess(self, inputs: Dict[str, Any]) -> Any:
        """Preprocess image for classification."""
        image_path = inputs.get('image_path')
        if not image_path:
            raise ValueError("image_path is required")

        # Load and preprocess image
        if load_image_from_path is None:
            raise ImportError("load_image_from_path utility not available")
        image = load_image_from_path(image_path)

        # Apply transforms if available
        if hasattr(self, 'transform') and self.transform:
            return self.transform(image).unsqueeze(0).to(self.device)

        return image

    def inference(self, processed_inputs: Any) -> Any:
        """Run classification inference."""
        if self.model is None:
            raise ValueError("Model not loaded")

        import torch
        with torch.no_grad():
            return self.model(processed_inputs)

    def postprocess(self, raw_outputs: Any) -> Dict[str, Any]:
        """Postprocess classification outputs."""
        import torch
        import torch.nn as nn

        if isinstance(raw_outputs, torch.Tensor):
            if self.task == 'fundus2age':
                # Regression task
                predicted_value = raw_outputs.cpu().item()
                return {
                    'prediction': round(predicted_value, 2),
                    'predictions': [f"{round(predicted_value, 1)}"]
                }
            else:
                # Classification task
                scores = nn.Sigmoid()(raw_outputs).cpu().numpy()[0]
                all_pairs = list(zip(self.classes, scores))

                # Filter by threshold
                filtered = [(cls, score) for cls, score in all_pairs if score >= self.threshold]
                if not filtered:
                    # Return top prediction if none above threshold
                    filtered = [max(all_pairs, key=lambda x: x[1])]

                return {
                    'probabilities': {cls: round(float(score), 3) for cls, score in filtered},
                    'predictions': [max(filtered, key=lambda x: x[1])[0]]
                }

        return raw_outputs

    def load_model(self):
        """Load classification model."""
        # Default implementation - override in subclasses
        logger.info(f"Loading {self.model_type} model for {self.name}")
        # This would be implemented by specific model loading logic

    def setup_preprocessing(self):
        """Setup preprocessing transforms."""
        # Default implementation - override in subclasses
        logger.info(f"Setting up preprocessing for {self.name}")
        # This would be implemented by specific preprocessing setup


class DetectionTool(ToolBase):
    """
    Base class for detection tools.

    Provides common functionality for object detection tasks.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.model_path = config.get('model_path', 'default')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.classes = config.get('classes', [])

    def execute(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Execute object detection on an image.

        Args:
            image_path: Path to the image

        Returns:
            List of detection results
        """
        results, metadata = self.run(image_path=image_path)
        return results

    def preprocess(self, inputs: Dict[str, Any]) -> Any:
        """Preprocess image for detection."""
        image_path = inputs.get('image_path')
        if not image_path:
            raise ValueError("image_path is required")

        if load_image_from_path is None:
            raise ImportError("load_image_from_path utility not available")
        image = load_image_from_path(image_path)

        # Apply transforms if available
        if hasattr(self, 'transform') and self.transform:
            return self.transform(image).unsqueeze(0).to(self.device)

        return image

    def inference(self, processed_inputs: Any) -> Any:
        """Run detection inference."""
        if self.model is None:
            raise ValueError("Model not loaded")

        import torch
        with torch.no_grad():
            return self.model(processed_inputs)

    def postprocess(self, raw_outputs: Any) -> List[Dict[str, Any]]:
        """Postprocess detection outputs."""
        # Default implementation - override in subclasses
        # This would parse bounding boxes, classes, and confidence scores
        return [
            {
                'class': 'detected_object',
                'bbox': [0, 0, 100, 100],
                'confidence': 0.9
            }
        ]

    def load_model(self):
        """Load detection model."""
        logger.info(f"Loading detection model for {self.name}")

    def setup_preprocessing(self):
        """Setup preprocessing transforms."""
        logger.info(f"Setting up preprocessing for {self.name}")


class SegmentationTool(ToolBase):
    """
    Base class for segmentation tools.

    Provides common functionality for image segmentation tasks.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.model_path = config.get('model_path', 'default')
        self.classes = config.get('classes', [])

    def execute(self, image_path: str) -> Dict[str, Any]:
        """
        Execute image segmentation.

        Args:
            image_path: Path to the image

        Returns:
            Segmentation result
        """
        results, metadata = self.run(image_path=image_path)
        return results

    def preprocess(self, inputs: Dict[str, Any]) -> Any:
        """Preprocess image for segmentation."""
        image_path = inputs.get('image_path')
        if not image_path:
            raise ValueError("image_path is required")

        if load_image_from_path is None:
            raise ImportError("load_image_from_path utility not available")
        image = load_image_from_path(image_path)

        # Apply transforms if available
        if hasattr(self, 'transform') and self.transform:
            return self.transform(image).unsqueeze(0).to(self.device)

        return image

    def inference(self, processed_inputs: Any) -> Any:
        """Run segmentation inference."""
        if self.model is None:
            raise ValueError("Model not loaded")

        import torch
        with torch.no_grad():
            return self.model(processed_inputs)

    def postprocess(self, raw_outputs: Any) -> Dict[str, Any]:
        """Postprocess segmentation outputs."""
        # Default implementation - override in subclasses
        return {
            'mask': 'base64_encoded_mask',
            'classes': self.classes,
            'confidence': 0.9
        }

    def load_model(self):
        """Load segmentation model."""
        logger.info(f"Loading segmentation model for {self.name}")

    def setup_preprocessing(self):
        """Setup preprocessing transforms."""
        logger.info(f"Setting up preprocessing for {self.name}")


class GenerationTool(ToolBase):
    """
    Base class for generation tools.

    Provides common functionality for content generation tasks.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.model_path = config.get('model_path', 'default')
        self.model_type = config.get('model_type', 'default')

    def execute(self, **kwargs) -> Any:
        """
        Execute content generation.

        Args:
            **kwargs: Generation parameters

        Returns:
            Generated content
        """
        results, metadata = self.run(**kwargs)
        return results

    def preprocess(self, inputs: Dict[str, Any]) -> Any:
        """Preprocess inputs for generation."""
        # Default implementation - override in subclasses
        return inputs

    def inference(self, processed_inputs: Any) -> Any:
        """Run generation inference."""
        if self.model is None:
            raise ValueError("Model not loaded")

        # Default implementation - override in subclasses
        return processed_inputs

    def postprocess(self, raw_outputs: Any) -> Any:
        """Postprocess generation outputs."""
        # Default implementation - override in subclasses
        return raw_outputs

    def load_model(self):
        """Load generation model."""
        logger.info(f"Loading generation model for {self.name}")

    def setup_preprocessing(self):
        """Setup preprocessing transforms."""
        logger.info(f"Setting up preprocessing for {self.name}")


class MultimodalTool(ToolBase):
    """
    Base class for multimodal tools.

    Provides common functionality for multimodal analysis tasks.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

    def execute(self, **kwargs) -> Any:
        """
        Execute multimodal analysis.

        Args:
            **kwargs: Analysis parameters

        Returns:
            Analysis result
        """
        results, metadata = self.run(**kwargs)
        return results

    def preprocess(self, inputs: Dict[str, Any]) -> Any:
        """Preprocess multimodal inputs."""
        # Default implementation - override in subclasses
        return inputs

    def inference(self, processed_inputs: Any) -> Any:
        """Run multimodal inference."""
        if self.model is None:
            raise ValueError("Model not loaded")

        # Default implementation - override in subclasses
        return processed_inputs

    def postprocess(self, raw_outputs: Any) -> Any:
        """Postprocess multimodal outputs."""
        # Default implementation - override in subclasses
        return raw_outputs

    def load_model(self):
        """Load multimodal model."""
        logger.info(f"Loading multimodal model for {self.name}")

    def setup_preprocessing(self):
        """Setup preprocessing transforms."""
        logger.info(f"Setting up preprocessing for {self.name}")

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class ToolBase(ABC):
    """
    Base class for all tools in EyeAgent.

    Provides common interface and functionality for tool registration,
    configuration, and execution with standardized lifecycle.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the tool.

        Args:
            name: Unique name for the tool
            config: Configuration dictionary for the tool
        """
        self.name = name
        self.config = config or {}
        self.enabled = True
        self.model = None
        self.device = self.config.get('device', 'cuda')
        self.model_path = self.config.get('model_path', None)
        self.initialized = False

        logger.info(f"Initialized tool: {name}")

    @abstractmethod
    def get_description(self) -> str:
        """
        Get tool description for documentation.

        Returns:
            Tool description string
        """
        pass

    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for tool inputs.

        Returns:
            Input schema dictionary
        """
        pass

    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for tool outputs.

        Returns:
            Output schema dictionary
        """
        pass

    def load_model(self) -> None:
        """
        Load the model for this tool.

        This method should be overridden by subclasses to implement
        specific model loading logic.
        """
        if self.model_path and Path(self.model_path).exists():
            logger.info(f"Loading model from {self.model_path}")
            # Default implementation - subclasses should override
            pass
        else:
            logger.warning(f"Model path not found: {self.model_path}")

    def preprocess(self, inputs: Dict[str, Any]) -> Any:
        """
        Preprocess inputs before inference.

        Args:
            inputs: Raw input data

        Returns:
            Preprocessed data ready for inference
        """
        # Default implementation - subclasses should override
        return inputs

    @abstractmethod
    def inference(self, processed_inputs: Any) -> Any:
        """
        Execute inference with the loaded model.

        Args:
            processed_inputs: Preprocessed input data

        Returns:
            Raw inference results
        """
        pass

    def postprocess(self, raw_outputs: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess raw inference outputs.

        Args:
            raw_outputs: Raw outputs from inference
            inputs: Original inputs for context

        Returns:
            Processed results dictionary
        """
        # Default implementation - subclasses should override
        return {"results": raw_outputs}

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Main execution method following the standard lifecycle:
        1. Initialize if needed
        2. Preprocess inputs
        3. Run inference
        4. Postprocess results

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Processed results dictionary
        """
        start_time = time.time()

        try:
            # Initialize model if not already done
            if not self.initialized:
                self.load_model()
                self.initialized = True

            # Preprocess inputs
            processed_inputs = self.preprocess(kwargs)

            # Run inference
            raw_outputs = self.inference(processed_inputs)

            # Postprocess results
            results = self.postprocess(raw_outputs, kwargs)

            # Add metadata
            results["_metadata"] = {
                "tool_name": self.name,
                "execution_time": time.time() - start_time,
                "status": "success"
            }

            logger.info(f"Tool {self.name} executed successfully in {results['_metadata']['execution_time']:.3f}s")
            return results

        except Exception as e:
            logger.error(f"Tool {self.name} execution failed: {e}")
            return {
                "error": str(e),
                "_metadata": {
                    "tool_name": self.name,
                    "execution_time": time.time() - start_time,
                    "status": "failed"
                }
            }

    def enable(self):
        """Enable the tool."""
        self.enabled = True
        logger.info(f"Enabled tool: {self.name}")

    def disable(self):
        """Disable the tool."""
        self.enabled = False
        logger.info(f"Disabled tool: {self.name}")

    def is_enabled(self) -> bool:
        """Check if tool is enabled."""
        return self.enabled

    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate input parameters against schema.

        Args:
            inputs: Input parameters to validate

        Returns:
            True if inputs are valid
        """
        # Basic validation - subclasses can override for more complex validation
        schema = self.get_input_schema()
        required_fields = schema.get('required', [])

        for field in required_fields:
            if field not in inputs:
                logger.error(f"Missing required field: {field}")
                return False

        return True

