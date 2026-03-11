会话管理方案（重构版）

1. 匹配策略
- 输入请求先编码为 token_ids。
- 在 KV 池中执行“最长 token 前缀匹配”，返回命中 block 链。
- 匹配基于 token，不基于原始文本字符串。

2. KV Cache 块模型
- 每块固定 block_size。
- 每块字段：block_id、parent_id、tokens、kv_ptr、ref_count、last_access、sealed。
- sealed=true 表示满块不可继续写；未满块允许追加。

3. 构建与复用流程
- 命中链后，链上块 ref_count += 1。
- 未命中的 token 后缀做增量 prefill，按 block_size 切块入池并挂接 parent。
- 生成阶段优先复用命中链，减少重复 prefill。

4. 引用与释放规则
- 上下文结束、替换或被新链覆盖时：旧链块 ref_count -= 1。
- ref_count == 0 的块进入可回收集合。
- 只有 ref_count == 0 才允许物理释放。

5. 容量与淘汰策略
- 设置 max_blocks / max_bytes 上限。
- 超限时，仅淘汰 ref_count == 0 的冷块（按 last_access 的 LRU）。
- 淘汰后同步更新索引，避免悬挂引用。

6. 并发与一致性
- 池操作统一加锁，ref_count 更新原子化。
- 先加引用再返回命中结果，避免并发释放。
- 发生异常时保证引用回滚，防止泄漏。

7. 异常回滚约束（必须）
- 任何请求在“已加引用但未完成建链”阶段失败，必须执行 ref_count 回滚。
- 建块失败时要清理本次新建的临时块与索引，再返回错误。
- 回滚流程需幂等：重复执行不会导致 ref_count 负数。

8. 未满块共享约束（必须）
- 默认只允许共享 sealed=true（满块）的块。
- sealed=false 的块仅允许被当前活跃上下文继续追加，不允许跨上下文复用。
- 当块写满后再转 sealed=true，才可进入共享索引。

9. 块 ID 生命周期约束（防 ABA）
- block_id 必须全局单调递增，不复用已删除 ID。
- 索引中保存 block_id 的同时保存 generation/version（可选但建议）。
- 命中后再次校验块存在性与状态，避免命中已回收后重建的新块。
