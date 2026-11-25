import torch

from layers.sail_image_memory import SAILImageMemoryFusion


def _views(count, batch, channels, height, width):
    return [
        {"res2": torch.randn(batch, channels, height, width)}
        for _ in range(count)
    ]


def test_anchor_query_tail_streaming_matches_dense():
    torch.manual_seed(7)
    batch, channels, height, width = 2, 16, 3, 4
    layer = SAILImageMemoryFusion(
        embed_dim=channels,
        num_heads=4,
        num_scales=1,
        tail_global_score_scale="res2",
        anchor_depth=1,
        dropout=0.0,
        alpha_init=0.05,
        tail_beta_init=0.01,
        attention_direction="anchor_to_tail",
    ).eval()
    tdom = [torch.randn(batch, channels, height, width)]
    anchors = _views(3, batch, channels, height, width)
    tails = _views(5, batch, channels, height, width)
    anchors[0]["res2"][0, :, 0, 0] = 0
    tails[1]["res2"][0, :, 1, 1] = 0
    anchor_exist = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    tail_exist = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 0, 1, 1, 0]], dtype=torch.bool
    )

    dense = layer(
        tdom,
        anchor_features=anchors,
        anchor_exist=anchor_exist,
        tail_features=tails,
        tail_exist=tail_exist,
    )
    chunks = [
        (tails[:2], tail_exist[:, :2]),
        (tails[2:4], tail_exist[:, 2:4]),
        (tails[4:], tail_exist[:, 4:]),
    ]
    streamed = layer.forward_streaming(
        tdom,
        anchor_features=anchors,
        anchor_exist=anchor_exist,
        tail_feature_chunks=iter(chunks),
    )
    torch.testing.assert_close(streamed[0], dense[0], rtol=2e-5, atol=2e-5)


def test_anchor_query_null_tail_streaming_matches_dense():
    torch.manual_seed(17)
    batch, channels, height, width = 2, 16, 3, 4
    layer = SAILImageMemoryFusion(
        embed_dim=channels,
        num_heads=4,
        num_scales=1,
        tail_global_score_scale="res2",
        anchor_depth=1,
        dropout=0.0,
        alpha_init=0.05,
        tail_beta_init=0.01,
        attention_direction="anchor_to_tail",
        use_null_tail=True,
    ).eval()
    tdom = [torch.randn(batch, channels, height, width)]
    anchors = _views(3, batch, channels, height, width)
    tails = _views(5, batch, channels, height, width)
    anchor_exist = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    tail_exist = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 0, 1, 1, 0]], dtype=torch.bool
    )
    dense = layer(
        tdom, anchor_features=anchors, anchor_exist=anchor_exist,
        tail_features=tails, tail_exist=tail_exist,
    )
    streamed = layer.forward_streaming(
        tdom,
        anchor_features=anchors,
        anchor_exist=anchor_exist,
        tail_feature_chunks=iter([
            (tails[:2], tail_exist[:, :2]),
            (tails[2:], tail_exist[:, 2:]),
        ]),
    )
    torch.testing.assert_close(streamed[0], dense[0], rtol=2e-5, atol=2e-5)


def test_tailwise_topk_streaming_matches_dense():
    torch.manual_seed(23)
    batch, channels, height, width = 2, 16, 3, 4
    layer = SAILImageMemoryFusion(
        embed_dim=channels,
        num_heads=4,
        num_scales=1,
        tail_global_score_scale="res2",
        anchor_depth=1,
        dropout=0.0,
        alpha_init=0.05,
        tail_beta_init=0.01,
        attention_direction="anchor_to_tail",
        tail_aggregation="tailwise_topk",
        tail_topk=3,
    ).eval()
    tdom = [torch.randn(batch, channels, height, width)]
    anchors = _views(3, batch, channels, height, width)
    tails = _views(5, batch, channels, height, width)
    anchor_exist = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    tail_exist = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 0, 1, 1, 0]], dtype=torch.bool
    )
    dense = layer(
        tdom, anchor_features=anchors, anchor_exist=anchor_exist,
        tail_features=tails, tail_exist=tail_exist,
    )
    streamed = layer.forward_streaming(
        tdom,
        anchor_features=anchors,
        anchor_exist=anchor_exist,
        tail_feature_chunks=iter([
            (tails[:1], tail_exist[:, :1]),
            (tails[1:3], tail_exist[:, 1:3]),
            (tails[3:], tail_exist[:, 3:]),
        ]),
    )
    torch.testing.assert_close(streamed[0], dense[0], rtol=2e-5, atol=2e-5)


def test_null_tail_keeps_anchor_only_when_no_real_tail_exists():
    torch.manual_seed(19)
    layer = SAILImageMemoryFusion(
        embed_dim=8,
        num_heads=2,
        num_scales=1,
        tail_global_score_scale="res2",
        attention_direction="anchor_to_tail",
        use_null_tail=True,
        tail_beta_init=0.1,
    ).eval()
    tdom = [torch.randn(1, 8, 2, 2)]
    anchors = _views(2, 1, 8, 2, 2)
    invalid_tails = _views(3, 1, 8, 2, 2)
    no_tail_output = layer(
        tdom, anchor_features=anchors,
        anchor_exist=torch.ones(1, 2, dtype=torch.bool),
    )
    null_output = layer(
        tdom, anchor_features=anchors,
        anchor_exist=torch.ones(1, 2, dtype=torch.bool),
        tail_features=invalid_tails,
        tail_exist=torch.zeros(1, 3, dtype=torch.bool),
    )
    torch.testing.assert_close(null_output[0], no_tail_output[0])


def test_anchor_query_tail_masks_do_not_create_nan():
    torch.manual_seed(11)
    layer = SAILImageMemoryFusion(
        embed_dim=8,
        num_heads=2,
        num_scales=1,
        tail_global_score_scale="res2",
        attention_direction="anchor_to_tail",
    ).eval()
    tdom = [torch.randn(1, 8, 2, 2)]
    anchors = _views(2, 1, 8, 2, 2)
    tails = _views(3, 1, 8, 2, 2)
    tail_exist = torch.zeros(1, 3, dtype=torch.bool)
    output = layer(
        tdom,
        anchor_features=anchors,
        anchor_exist=torch.ones(1, 2, dtype=torch.bool),
        tail_features=tails,
        tail_exist=tail_exist,
    )
    assert torch.isfinite(output[0]).all()


def test_original_direction_remains_default():
    layer = SAILImageMemoryFusion(embed_dim=8, num_heads=8, num_scales=1, tail_global_score_scale="res2")
    assert layer.attention_direction == "tail_to_anchor"


def test_cross_view_local_anchor_memory_is_order_equivariant_and_finite():
    """K-medoid order is arbitrary, so a spatial R encoder must not use it."""
    torch.manual_seed(29)
    batch, channels, height, width = 2, 16, 5, 7
    layer = SAILImageMemoryFusion(
        embed_dim=channels,
        num_heads=4,
        num_scales=1,
        tail_global_score_scale="res2",
        attention_direction="anchor_to_tail",
        tail_beta_init=0.01,
        anchor_spatial_window_size=5,
        anchor_spatial_scales=["res2"],
    ).eval()
    tdom = [torch.randn(batch, channels, height, width)]
    anchors = _views(5, batch, channels, height, width)
    tails = _views(3, batch, channels, height, width)
    anchor_exist = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.bool
    )
    tail_exist = torch.ones(batch, 3, dtype=torch.bool)
    anchors[0]["res2"][0, :, 0, 0] = 0

    output = layer(
        tdom, anchor_features=anchors, anchor_exist=anchor_exist,
        tail_features=tails, tail_exist=tail_exist,
    )
    permutation = torch.tensor([3, 1, 4, 0, 2])
    permuted_output = layer(
        tdom,
        anchor_features=[anchors[i] for i in permutation],
        anchor_exist=anchor_exist[:, permutation],
        tail_features=tails,
        tail_exist=tail_exist,
    )
    assert torch.isfinite(output[0]).all()
    torch.testing.assert_close(output[0], permuted_output[0], rtol=2e-5, atol=2e-5)


def test_cross_view_local_anchor_memory_has_backward_path():
    torch.manual_seed(31)
    layer = SAILImageMemoryFusion(
        embed_dim=8,
        num_heads=2,
        num_scales=1,
        tail_global_score_scale="res2",
        attention_direction="anchor_to_tail",
        tail_beta_init=0.01,
        anchor_spatial_window_size=3,
        anchor_spatial_scales=["res2"],
    )
    tdom = [torch.randn(1, 8, 4, 4, requires_grad=True)]
    anchors = _views(3, 1, 8, 4, 4)
    output = layer(
        tdom,
        anchor_features=anchors,
        anchor_exist=torch.ones(1, 3, dtype=torch.bool),
        tail_features=_views(2, 1, 8, 4, 4),
        tail_exist=torch.ones(1, 2, dtype=torch.bool),
    )
    output[0].square().mean().backward()
    assert layer.anchor_spatial_blocks[0][0].q_proj.weight.grad is not None
    assert torch.isfinite(layer.anchor_spatial_blocks[0][0].q_proj.weight.grad).all()


def test_cross_view_local_anchor_memory_keeps_all_tail_streaming_exact():
    torch.manual_seed(37)
    layer = SAILImageMemoryFusion(
        embed_dim=16,
        num_heads=4,
        num_scales=1,
        tail_global_score_scale="res2",
        attention_direction="anchor_to_tail",
        tail_beta_init=0.01,
        anchor_spatial_window_size=3,
        anchor_spatial_scales=["res2"],
    ).eval()
    tdom = [torch.randn(1, 16, 4, 5)]
    anchors = _views(5, 1, 16, 4, 5)
    tails = _views(6, 1, 16, 4, 5)
    anchor_exist = torch.ones(1, 5, dtype=torch.bool)
    tail_exist = torch.tensor([[1, 1, 0, 1, 1, 1]], dtype=torch.bool)
    dense = layer(
        tdom, anchor_features=anchors, anchor_exist=anchor_exist,
        tail_features=tails, tail_exist=tail_exist,
    )
    streamed = layer.forward_streaming(
        tdom,
        anchor_features=anchors,
        anchor_exist=anchor_exist,
        tail_feature_chunks=iter([
            (tails[:2], tail_exist[:, :2]),
            (tails[2:], tail_exist[:, 2:]),
        ]),
    )
    torch.testing.assert_close(streamed[0], dense[0], rtol=2e-5, atol=2e-5)


def test_tail_query_local_anchor_token_memory_streams_and_backpropagates():
    """SAIL-like query read: tail Q -> unpooled local anchor token set K/V."""
    torch.manual_seed(41)
    layer = SAILImageMemoryFusion(
        embed_dim=16,
        num_heads=4,
        num_scales=1,
        tail_global_score_scale="res2",
        attention_direction="tail_to_anchor",
        tail_beta_init=1.0,
        tail_anchor_memory_mode="local_token_set",
        tail_anchor_local_window_size=3,
        tail_anchor_local_scales=["res2"],
        memory_output_mode="tail_conditioned",
    )
    tdom = [torch.randn(2, 16, 4, 5)]
    anchors = _views(5, 2, 16, 4, 5)
    tails = _views(6, 2, 16, 4, 5)
    anchor_exist = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.bool
    )
    tail_exist = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 0]], dtype=torch.bool
    )
    dense = layer(
        tdom, anchor_features=anchors, anchor_exist=anchor_exist,
        tail_features=tails, tail_exist=tail_exist,
    )
    streamed = layer.forward_streaming(
        tdom,
        anchor_features=anchors,
        anchor_exist=anchor_exist,
        tail_feature_chunks=iter([
            (tails[:2], tail_exist[:, :2]),
            (tails[2:], tail_exist[:, 2:]),
        ]),
    )
    torch.testing.assert_close(streamed[0], dense[0], rtol=2e-5, atol=2e-5)
    dense[0].square().mean().backward()
    grad = layer.tail_anchor_local_blocks[0].q_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_r_conditioned_local_tail_attention_streams_and_backpropagates():
    """R(x,y) reads tail neighbourhoods; streaming remains exact."""
    torch.manual_seed(53)
    batch, channels, height, width = 2, 16, 5, 6
    layer = SAILImageMemoryFusion(
        embed_dim=channels,
        num_heads=4,
        num_scales=1,
        anchor_depth=1,
        dropout=0.0,
        attention_direction="anchor_to_tail",
        tail_beta_init=0.01,
        tail_global_score_scale="res2",
        r_conditioned_local_tail_enabled=True,
        r_conditioned_local_tail_scales=["res2"],
        r_conditioned_local_tail_window_sizes={"res2": 3},
        r_conditioned_local_tail_attention_dim=8,
        r_conditioned_local_tail_num_heads=2,
    ).eval()
    tdom = [torch.randn(batch, channels, height, width)]
    anchors = _views(3, batch, channels, height, width)
    tails = _views(5, batch, channels, height, width)
    anchors[0]["res2"][0, :, 0, 0] = 0
    tails[1]["res2"][0, :, 2, 2] = 0
    anchor_exist = torch.ones(batch, 3, dtype=torch.bool)
    tail_exist = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 0, 1, 1, 0]], dtype=torch.bool
    )

    dense = layer(
        tdom,
        anchor_features=anchors,
        anchor_exist=anchor_exist,
        tail_features=tails,
        tail_exist=tail_exist,
    )
    streamed = layer.forward_streaming(
        tdom,
        anchor_features=anchors,
        anchor_exist=anchor_exist,
        tail_feature_chunks=iter([
            (tails[:2], tail_exist[:, :2]),
            (tails[2:4], tail_exist[:, 2:4]),
            (tails[4:], tail_exist[:, 4:]),
        ]),
    )
    torch.testing.assert_close(streamed[0], dense[0], rtol=2e-5, atol=2e-5)
    assert torch.isfinite(dense[0]).all()

    dense[0].square().mean().backward()
    local_block = layer.r_conditioned_local_tail_blocks[0]
    assert local_block.q_proj.weight.grad is not None
    assert torch.isfinite(local_block.q_proj.weight.grad).all()
    # The replaced same-pixel MHA must not leave unused trainable parameters
    # behind, otherwise normal DDP would fail on the second iteration.
    unused = [
        name for name, parameter in layer.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert unused == []
