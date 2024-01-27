from collections import namedtuple

def wd_v14_swin2_tagger_config():
    CustomConfig = namedtuple('CustomConfig', [
        'architecture', 'num_classes', 'num_features', 'global_pool', 'model_args', 'pretrained_cfg'
    ])

    custom_config = CustomConfig(
        architecture="swinv2_base_window8_256",
        num_classes=9083,
        num_features=1024,
        global_pool="avg",
        model_args={
            "act_layer": "gelu",
            "img_size": 448,
            "window_size": 14
        },
        pretrained_cfg={
            "custom_load": False,
            "input_size": [3, 448, 448],
            "fixed_input_size": False,
            "interpolation": "bicubic",
            "crop_pct": 1.0,
            "crop_mode": "center",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "num_classes": 9083,
            "pool_size": None,
            "first_conv": None,
            "classifier": None
        }
    )
    return custom_config