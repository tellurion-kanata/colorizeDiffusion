def convert_sd_ckpt(sd):
    new_sd = {}
    for k in sd.keys():
        # if k.find("model.diffusion_model.middle_block.2.") > -1:
        #     new_sd[k.replace(
        #         "model.diffusion_model.middle_block.2.",
        #         "model.diffusion_model.middle_block.3."
        #     )] = sd[k].clone()
        # plus 1
        if k.find("model.diffusion_model.output_blocks.2.1.conv.") > -1:
            new_sd[k.replace(
                "model.diffusion_model.output_blocks.2.1.conv",
                "model.diffusion_model.output_blocks.3.0.conv"
            )] = sd[k].clone()

        # plus 1 layers
        elif k.find("model.diffusion_model.output_blocks.3") > -1:
            new_sd[k.replace(
                "model.diffusion_model.output_blocks.3",
                "model.diffusion_model.output_blocks.4"
            )] = sd[k].clone()
        elif k.find("model.diffusion_model.output_blocks.4") > -1:
            new_sd[k.replace(
                "model.diffusion_model.output_blocks.4",
                "model.diffusion_model.output_blocks.5"
            )] = sd[k].clone()
        elif k.find("model.diffusion_model.output_blocks.5") > -1 and k.find("conv") == -1:
            new_sd[k.replace(
                "model.diffusion_model.output_blocks.5",
                "model.diffusion_model.output_blocks.6"
            )] = sd[k].clone()

        # plus 2
        elif k.find("model.diffusion_model.output_blocks.5.2.conv.") > -1:
            new_sd[k.replace(
                "model.diffusion_model.output_blocks.5.2.conv",
                "model.diffusion_model.output_blocks.7.0.conv"
            )] = sd[k].clone()

        # plus 2 layers
        elif k.find("model.diffusion_model.output_blocks.6") > -1 and k.find("conv") == -1:
            new_sd[k.replace(
                "model.diffusion_model.output_blocks.6",
                "model.diffusion_model.output_blocks.8"
            )] = sd[k].clone()
        elif k.find("model.diffusion_model.output_blocks.7") > -1:
            new_sd[k.replace(
                "model.diffusion_model.output_blocks.7",
                "model.diffusion_model.output_blocks.9"
            )] = sd[k].clone()
        elif k.find("model.diffusion_model.output_blocks.8") > -1 and k.find("conv") == -1:
            new_sd[k.replace(
                "model.diffusion_model.output_blocks.8",
                "model.diffusion_model.output_blocks.10"
            )] = sd[k].clone()

        # plus 3
        elif k.find("model.diffusion_model.output_blocks.8.2.conv.") > -1:
            new_sd[k.replace(
                "model.diffusion_model.output_blocks.8.2.conv",
                "model.diffusion_model.output_blocks.11.0.conv"
            )] = sd[k].clone()

        # plus 3 layers
        elif k.find("model.diffusion_model.output_blocks.9") > -1:
            new_sd[k.replace(
                "model.diffusion_model.output_blocks.9",
                "model.diffusion_model.output_blocks.12"
            )] = sd[k].clone()
        elif k.find("model.diffusion_model.output_blocks.10") > -1:
            new_sd[k.replace(
                "model.diffusion_model.output_blocks.10",
                "model.diffusion_model.output_blocks.13"
            )] = sd[k].clone()
        elif k.find("model.diffusion_model.output_blocks.11") > -1:
            new_sd[k.replace(
                "model.diffusion_model.output_blocks.11",
                "model.diffusion_model.output_blocks.14"
            )] = sd[k].clone()

        else:
            new_sd[k] = sd[k].clone()
    return new_sd