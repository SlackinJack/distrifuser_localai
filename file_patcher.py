root = "DistriFuser/distrifuser"
models = f"{root}/models"
modules = f"{root}/modules"

files_to_patch = [
    # models
    {
        "file_name": f"{models}/base_model.py",
        "replace": [
            {
                "from": "from distrifuser.modules",
                "to":   "from ..modules",
            },
        ],
    },
    {
        "file_name": f"{models}/distri_sdxl_unet_pp.py",
        "replace": [
            {
                "from": "from distrifuser.modules",
                "to":   "from ..modules",
            },
        ],
    },
    {
        "file_name": f"{models}/distri_sdxl_unet_tp.py",
        "replace": [
            {
                "from": "from distrifuser.modules",
                "to":   "from ..modules",
            },
        ],
    },
    # modules/pp
    {
        "file_name": f"{modules}/pp/attn.py",
        "replace": [
            {
                "from": "from distrifuser.modules.base_module",
                "to":   "from ..base_module",
            },
            {
                "from": "from distrifuser.utils",
                "to":   "from ...utils",
            },
        ],
    },
    {
        "file_name": f"{modules}/pp/conv2d.py",
        "replace": [
            {
                "from": "from distrifuser.modules.base_module",
                "to":   "from ..base_module",
            },
            {
                "from": "from distrifuser.utils",
                "to":   "from ...utils",
            },
        ],
    },
    {
        "file_name": f"{modules}/pp/groupnorm.py",
        "replace": [
            {
                "from": "from distrifuser.modules.base_module",
                "to":   "from ..base_module",
            },
            {
                "from": "from distrifuser.utils",
                "to":   "from ...utils",
            },
        ],
    },
    # modules/tp
    {
        "file_name": f"{modules}/tp/attention.py",
        "replace": [
            {
                "from": "from distrifuser.modules.base_module",
                "to":   "from ..base_module",
            },
            {
                "from": "from distrifuser.utils",
                "to":   "from ...utils",
            },
        ],
    },
    {
        "file_name": f"{modules}/tp/conv2d.py",
        "replace": [
            {
                "from": "from distrifuser.modules.base_module",
                "to":   "from ..base_module",
            },
            {
                "from": "from distrifuser.utils",
                "to":   "from ...utils",
            },
        ],
    },
    # modules
    {
        "file_name": f"{modules}/base_module.py",
        "replace": [
            {
                "from": "from distrifuser.utils",
                "to":   "from ..utils",
            },
        ],
    },
]


for f in files_to_patch:
    file_name = f.get("file_name")
    try:
        print(f"Patching file: {file_name}")
        with open(file_name, "r") as file:              data = file.read()                                  # read file
        with open(file_name + ".bak", "w") as file:     file.write(data)                                    # create backup
        for r in f.get("replace"):                      data = data.replace(r.get("from"), r.get("to"))     # replace strings
        with open(file_name, "w") as file:              file.write(data)                                    # overwrite file
        print(f"File patched: {file_name}")
    except Exception as e:
        print(f"Failed to patch file: {file_name}")
        print(str(e))

