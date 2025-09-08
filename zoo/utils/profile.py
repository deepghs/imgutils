import torch
from ditk import logging
from thop import profile, clever_format


def torch_model_profile_via_thop(model, input_):
    with torch.no_grad():
        flops, params = profile(model, (input_,))

    s_flops, s_params = clever_format([flops, params], "%.1f")
    logging.info(f'Params: {s_params}, FLOPs: {s_flops}.')

    return flops, params


def torch_model_profile_via_calflops(model, input_):
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=tuple(input_.shape),
        output_as_string=False,
        print_detailed=False,
        # output_as_string=True,
        # output_precision=4
    )
    s_flops, s_params, s_macs = clever_format([flops, params, macs], "%.1f")
    logging.info(f'Params: {s_params}, FLOPs: {s_flops}, MACs: {s_macs}.')
    return flops, params, macs


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    from timm import create_model

    # model = create_model('hf-hub:animetimm/swinv2_base_window8_256.dbv4-full', pretrained=False)
    model = create_model('caformer_b36.sail_in22k_ft_in1k_384', pretrained=False)
    dummy_input = torch.randn(1, 3, 448, 448)
    print(torch_model_profile_via_thop(model, dummy_input))
    print(torch_model_profile_via_calflops(model, dummy_input))
