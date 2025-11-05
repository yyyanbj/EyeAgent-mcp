from typing import Dict, Type, Any

# Global registry for agents
AGENT_REGISTRY: Dict[str, Type] = {}


def register_agent(name: str):
    """Class decorator to register an agent by name.

    Usage:
        @register_agent("knowledge")
        class KnowledgeAgent(Agent):
            ...
    """
    def decorator(cls: Type) -> Type:
        AGENT_REGISTRY[name] = cls
        cls.__agent_name__ = name
        return cls
    return decorator


def get_agent_class(name: str):
    return AGENT_REGISTRY.get(name)


def create_agent(name: str, **kwargs: Any):
    cls = get_agent_class(name)
    if cls is None:
        raise ValueError(f"Unknown agent: {name}")
    return cls(**kwargs)
