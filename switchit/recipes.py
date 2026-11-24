"""Immutable, content-addressed harness recipes with pinned runtime versions."""
from copy import deepcopy
from pathlib import Path
import json
from switchit.contracts import fingerprint, require, version
from switchit.storage import atomic_write_json


def validate_recipe(recipe):
    version(recipe, 'recipe_version')
    for field in ('name', 'runtime', 'runtime_version', 'workflow_revision'):
        require(isinstance(recipe.get(field), str) and bool(recipe[field]), f'{field} is required')
    require(isinstance(recipe.get('allowed_tools'), list) and all(isinstance(t, str) for t in recipe['allowed_tools']), 'allowed_tools must be explicit')
    require(set(recipe) <= {'recipe_version', 'name', 'runtime', 'runtime_version', 'workflow_revision',
                          'model', 'system_prompt', 'allowed_tools', 'config', 'recipe_id'}, 'unknown recipe field')
    require(isinstance(recipe.get('config', {}), dict), 'config must be an object')
    require(recipe.get('recipe_id') == recipe_id(recipe), 'recipe digest mismatch')
    return recipe


def recipe_id(recipe):
    return fingerprint({k: v for k, v in recipe.items() if k != 'recipe_id'})


def make_recipe(name, runtime, runtime_version, workflow_revision, allowed_tools, **options):
    recipe = {'recipe_version': 1, 'name': name, 'runtime': runtime, 'runtime_version': runtime_version,
              'workflow_revision': workflow_revision, 'allowed_tools': list(allowed_tools), **deepcopy(options)}
    recipe['recipe_id'] = recipe_id(recipe)
    return validate_recipe(recipe)


class RecipeRegistry:
    def __init__(self, root):
        self.root = Path(root)

    def put(self, recipe):
        validate_recipe(recipe)
        path = self.root / (recipe['recipe_id'] + '.json')
        if path.exists():
            require(json.loads(path.read_text()) == recipe, 'immutable recipe collision')
        else:
            atomic_write_json(path, recipe)
        return recipe['recipe_id']

    def get(self, key):
        require(len(key) == 64 and all(c in '0123456789abcdef' for c in key), 'invalid recipe ID')
        recipe = json.loads((self.root / (key + '.json')).read_text())
        validate_recipe(recipe)
        require(recipe['recipe_id'] == key, 'registry content mismatch')
        return recipe
