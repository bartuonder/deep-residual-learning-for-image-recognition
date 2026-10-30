import pytest
import torch

from resnet.blocks import BasicBlock, Bottleneck, PaddedIdentity, Projection, make_shortcut
from resnet.cifar import cifar_resnet
from resnet.complexity import count_flops, count_parameters
from resnet.imagenet import imagenet_resnet

# Table 1 of the paper (FLOPs = multiply-adds, in units of 1e9)
IMAGENET_FLOPS = {18: 1.8, 34: 3.6, 50: 3.8, 101: 7.6, 152: 11.3}

# Fig. 6 / Sec. 4.2: parameter counts of the CIFAR-10 nets (millions)
CIFAR_PARAMS = {20: 0.27, 32: 0.46, 44: 0.66, 56: 0.85, 110: 1.7}


@pytest.mark.parametrize("depth,expected", IMAGENET_FLOPS.items())
def test_imagenet_flops_match_table1(depth, expected):
    flops = count_flops(imagenet_resnet(depth)) / 1e9
    assert flops == pytest.approx(expected, abs=0.1)


@pytest.mark.parametrize("depth,expected", CIFAR_PARAMS.items())
def test_cifar_params_match_paper(depth, expected):
    params = count_parameters(cifar_resnet(depth)) / 1e6
    assert params == pytest.approx(expected, rel=0.03)


def test_cifar_1202_params():
    assert count_parameters(cifar_resnet(1202)) / 1e6 == pytest.approx(19.4, rel=0.01)


def test_plain_and_residual_cifar_have_same_params_with_option_a():
    # Sec. 4.2: option A makes residual nets identical to plain nets in
    # depth, width and number of parameters.
    for depth in (20, 56):
        assert count_parameters(cifar_resnet(depth)) == \
            count_parameters(cifar_resnet(depth, residual=False))


def test_plain34_and_resnet34_cost_the_same():
    # Fig. 3: the plain and residual 34-layer nets are both 3.6 billion FLOPs
    plain = count_flops(imagenet_resnet(34, residual=False)) / 1e9
    resnet = count_flops(imagenet_resnet(34)) / 1e9
    assert plain == pytest.approx(3.6, abs=0.1)
    assert resnet == pytest.approx(3.6, abs=0.1)


def test_forward_shapes():
    assert cifar_resnet(20)(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert imagenet_resnet(18)(torch.randn(2, 3, 224, 224)).shape == (2, 1000)


@pytest.mark.parametrize("option,expected", [("A", PaddedIdentity), ("B", Projection), ("C", Projection)])
def test_shortcut_types_on_dimension_change(option, expected):
    assert isinstance(make_shortcut(option, 16, 32, 2), expected)


def test_shortcut_on_same_shape():
    assert isinstance(make_shortcut("A", 16, 16, 1), torch.nn.Identity)
    assert isinstance(make_shortcut("B", 16, 16, 1), torch.nn.Identity)
    assert isinstance(make_shortcut("C", 16, 16, 1), Projection)


def test_padded_identity_is_parameter_free_and_zero_pads():
    sc = PaddedIdentity(4, 8, stride=2)
    x = torch.randn(1, 4, 8, 8)
    y = sc(x)
    assert y.shape == (1, 8, 4, 4)
    assert torch.equal(y[:, :4], x[:, :, ::2, ::2])
    assert torch.count_nonzero(y[:, 4:]) == 0
    assert count_parameters(sc) == 0


def test_residual_block_adds_input():
    block = BasicBlock(8, 8).eval()
    # with the second BN scaled to zero, F(x) = 0 and the block is relu(x)
    torch.nn.init.zeros_(block.bn2.weight)
    torch.nn.init.zeros_(block.bn2.bias)
    x = torch.randn(2, 8, 6, 6)
    assert torch.allclose(block(x), torch.relu(x))


def test_bottleneck_stride_placement():
    b1 = Bottleneck(256, 128, stride=2)
    b2 = Bottleneck(256, 128, stride=2, stride_on_3x3=True)
    assert b1.conv1.stride == (2, 2) and b1.conv2.stride == (1, 1)
    assert b2.conv1.stride == (1, 1) and b2.conv2.stride == (2, 2)
