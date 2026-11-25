# Release preparation

- Selected the project entry points and their local source dependencies from the research tree.
- Added Chinese README, source navigation, input requirements and test runners.
- Replaced machine-specific examples with generic paths and kept upstream notices.
- Kept the original algorithm and historical results unless an explicit fix is listed below.
- Original server projects were not modified. Full benchmark runs were not repeated.

- Existing single-scale test fixtures now explicitly select `tail_global_score_scale="res2"`, matching their sole available feature scale.
- Fixed the null-tail single-query branch: normalize the `[N, 1]` query-validity mask to `[N]` before broadcasting to `[N, heads, tails]`. Previously the mask introduced an extra axis and failed in einsum. Existing null-tail and dense/streaming tests now pass.
- No new segmentation-accuracy claim is made for this source fix.
