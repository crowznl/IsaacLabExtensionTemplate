# 0. Now you can directly fork&clone the Zbot branch of upstream repo by acessing the following link:
>[Zbot/IsaacLabExtensionTemplate](https://github.com/crowznl/IsaacLabExtensionTemplate)
>
>**If you do so, you can skip the following steps :)**
---

# 1. clone
[IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate)

```
git clone https://github.com/isaac-sim/IsaacLabExtensionTemplate.git
```

## 1.1 rename
```python
# Enter the repository

cd IsaacLabExtensionTemplate

# Rename all occurrences of ext_template (in files/directories) to your_fancy_extension_name (Zbot for example)

python scripts/rename_template.py Zbot
```
Then, you can put my 'Zbot'package into `IsaacLabExtensionTemplate/exts/` to replace the same name folder

>otherwise, you can manually rename package name in 
>
>`IsaacLabExtensionTemplate/scripts/rsl_rl/play.py` line53 and `IsaacLabExtensionTemplate/scripts/rsl_rl/train.py` line68 to
>
>`import Zbot.tasks`
>
>Then, you can put my 'Zbot'package into `IsaacLabExtensionTemplate/exts/` to replace the 'ext_template' folder

## 1.2 install ***
```python
# Using a python interpreter that has Isaac Lab installed, install the library
python -m pip install -e exts/Zbot
```
# 2.set up IDE 
follow the instruction in [IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate)

>Check settings.json in the .vscode directory

```json
    // Python extra paths
    "python.analysis.extraPaths": [
        "~/IsaacLab/source/extensions/omni.isaac.lab_tasks",
        "~/IsaacLab/source/extensions/omni.isaac.lab",
        "~/IsaacLab/source/extensions/omni.isaac.lab_assets"
    ]
```

# 3.assets
in `IsaacLabExtensionTemplate/exts/Zbot/Zbot/assets/__init__.py`
```python
ISAACLAB_ASSETS_DATA_DIR = "/home/crowznl/Dev/isaac/asset/zbot/"
```
change the path to your zbot assets path
>Attention: 
>
>the path should be absolute path. Do not use `~`.

