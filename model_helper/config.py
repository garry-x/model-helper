"""Provider configuration management.

Stores the list of configured providers in ~/.model_helper/providers.json.
When providers are configured, only models from those providers are cached/displayed.

If the user config file does not exist, defaults are read from
model_helper/default_providers.json (shipped with the project, can be
edited or version-controlled).
"""

import json
from pathlib import Path


def _defaults_path() -> Path:
    """Path to the shipped default-providers config file."""
    return Path(__file__).parent / "default_providers.json"


def _config_path() -> Path:
    """Path to the user's provider config."""
    p = Path.home() / ".model_helper" / "providers.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_defaults() -> tuple[list[str], dict[str, list[str]]]:
    """Load default providers and aliases from the shipped config file."""
    path = _defaults_path()
    if not path.exists():
        return [], {}
    try:
        data = json.loads(path.read_text())
        providers = [p.lower() for p in data.get("providers", [])]
        aliases = {k.lower(): [a.lower() for a in v] for k, v in data.get("aliases", {}).items()}
        return providers, aliases
    except (json.JSONDecodeError, ValueError):
        return [], {}


def get_providers() -> list[str]:
    """Return the list of configured providers.
    Falls back to defaults (from default_providers.json) if no user config exists.
    """
    path = _config_path()
    if not path.exists():
        return _load_defaults()[0]
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return [p.lower() for p in data]
    except (json.JSONDecodeError, ValueError):
        pass
    return _load_defaults()[0]


def save_providers(providers: list[str]) -> None:
    """Save the provider list to user config."""
    _config_path().write_text(json.dumps(providers, indent=2))


def add_provider(name: str) -> bool:
    """Add a provider. Returns True if added, False if already exists."""
    providers = get_providers()
    name_lower = name.lower()
    if name_lower in providers:
        return False
    providers.append(name_lower)
    save_providers(providers)
    return True


def remove_provider(name: str) -> bool:
    """Remove a provider. Returns True if removed, False if not found."""
    providers = get_providers()
    name_lower = name.lower()
    if name_lower not in providers:
        return False
    providers.remove(name_lower)
    save_providers(providers)
    return True


def resolve_providers(providers: list[str]) -> list[str]:
    """Expand provider names to include aliases for matching against cache data.

    Aliases are read from default_providers.json (the 'aliases' section).
    E.g. ['alibaba'] -> ['alibaba', 'dashscope']
         ['bytedance'] -> ['bytedance', 'volcengine']
    """
    _, alias_map = _load_defaults()
    result = list(providers)
    for p in providers:
        for alias in alias_map.get(p, []):
            if alias not in result:
                result.append(alias)
    return result
