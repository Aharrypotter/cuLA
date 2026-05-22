# Plan

这个目录记录任务目标、阶段拆分、验证矩阵、风险和验收标准。

## Documents

- [issue10-target-spec.md](./issue10-target-spec.md): issue #10 的目标规格。明确 larger chunk 是交付目标，cooperative 2-CTA 是支撑 `chunk_size>=128` 的手段；后续 protocol 工作必须回连到这个目标。
- [issue-10-focus.md](./issue-10-focus.md): issue #10 的短执行入口，聚焦当前任务、下一步命令和决策点。
- [issue-10-sm10x-chunk-2cta-plan.md](./issue-10-sm10x-chunk-2cta-plan.md): issue #10 的 larger chunk / 2-CTA 计划，包括当前仓库映射、patch 状态、推荐执行顺序、验证覆盖和风险缓解。
- [issue-10-2cta-cluster-protocol-plan.md](./issue-10-2cta-cluster-protocol-plan.md): 2-CTA cluster protocol deadlock 修复的分阶段执行计划（Stage 1 = 最小 `cta_layout_vmnk` + `is_leader_cta` gating；Stage 2 = bisection；Stage 3 = QK N-half；Stage 4 = perf）。
- [gdn-kernel-migration-plan.md](./gdn-kernel-migration-plan.md): GDN kernel 迁移到 cuLA 的 SM100/B200-first 分阶段计划；以 FlashInfer PR #3001 Blackwell CuTe DSL chunked prefill 为首个迁移目标，并记录 PR #2276 SM90 路径作为后续可选 backend。

## Scope

- 放下一步工作、阶段状态和验收标准。
- 已经发生的日期型进展放到 `../mainline/`。
- 方案设计细节放到 `../design/`。
- 环境操作和 Modal 用法放到 `../env/`。
