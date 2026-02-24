import os
import json
from pathlib import Path
from ..seed_gen.parsing import parse_schema
from ..seed_gen.models.tool_schemas import ToolSchema

def load_schemas(schemas_dir: str):
    """
    Helper to load schemas from directory using unified multi-format parser.
    Supports .json, .yaml, .yml, and .txt (Natural Language descriptions).
    """
    schemas = []
    if os.path.exists(schemas_dir):
        for f_name in sorted(os.listdir(schemas_dir)):
            if f_name.endswith((".json", ".yaml", ".yml", ".txt")):
                f_path = Path(schemas_dir) / f_name
                try:
                    with open(f_path, "r") as f:
                        if f_name.endswith((".yaml", ".yml")):
                            import yaml
                            input_data = yaml.safe_load(f)
                        elif f_name.endswith(".json"):
                            input_data = json.load(f)
                        else:  # .txt (NL)
                            input_data = f.read()
                    
                    canonical = parse_schema(input_data)
                    # Convert to ToolSchema for internal SeedGenerator compatibility
                    tool_schema = ToolSchema.from_canonical(canonical)
                    schemas.append(tool_schema)
                except Exception as e:
                    print(f"Warning: Failed to load schema {f_name}: {e}")
    else:
        print(f"Warning: Schemas dir {schemas_dir} not found.")
    return schemas
