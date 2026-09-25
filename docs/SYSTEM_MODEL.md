# 多微网协同调度系统建模说明书

> **用途**：供研究者阅读、介绍系统组成和理解当前数学模型；作为后续讨论环境设定时的共同建模基准。本文描述**已经实现的环境**，不将尚未实现的设想写作现有功能。  
> **代码基准**：仓库 `lmjaoa-bit/multi_microgrid`，分支 `代号：噜噜`，建模核对提交 `fa7cea4541bb82f759d0df9bcf2d1bdc7b5804ee`（2026-09-23）。  
> **适用范围**：`configs/cooperative.yaml` 的 `single_dispatch_dr` 协同环境，包含 MG1 工业磨机、MG2 商业 HVAC、MG3 住宅 EV，以及三个微网的 ESS 与 P2P 能量交易。文档不等同于独立的 AC 潮流模型，也不代表所有历史分支的建模设定。
>
> **公式阅读说明**：全文使用 Markdown + LaTeX：行间公式置于 `$$ ... $$` 中；功率单位 kW，能量单位 kWh，时间单位 h。建议在支持数学公式渲染的 Markdown 阅读器中查看。

## 1. 系统结构与统一记号

### 1.1 微网及设备组成

系统包含三个异构微网。MG1 配置光伏、储能和工业原料磨机；MG2 配置风电、储能和商业楼宇 HVAC；MG3 配置光伏、储能和聚合 EV 充电。各微网既可独立与公共电网交互，也可通过协同能源池开展微网间 P2P 交易。

| 组成 | MG1：工业 | MG2：商业 | MG3：住宅 |
|---|---:|---:|---:|
| 光伏装机容量 | 520 kW | 0 | 100 kW |
| 风电装机容量 | 0 | 900 kW | 0 |
| 储能容量 | 210 kWh | 290 kWh | 95 kWh |
| 储能最大充/放电功率 | 65/65 kW | 80/80 kW | 34/34 kW |
| 储能初始 SOC | 0.50 | 0.50 | 0.50 |
| 特有柔性设备 | 28.5 kW 磨机 | 最大制热/制冷电功率各 35 kW 的 HVAC | 780 kWh、93.6 kW 的 EV 车队 |
| 外部电网最大购/售电功率 | 200/200 kW | 200/200 kW | 200/200 kW |

每个**无序微网对**共用一条 75 kW 的 P2P 交易容量预算；同一对微网的两个方向以及不同来源、不同用途的交易，均占用该容量。当前配置 P2P 传输损耗为零。

### 1.2 时间、集合与符号

$$
\mathcal M=\{1,2,3\},\qquad
\mathcal T=\{0,1,\ldots,23\},\qquad
\Delta t=1\ {\rm h},\qquad T=24.
$$

令 `i,j\in\mathcal M` 表示微网，`t\in\mathcal T` 表示当前调度时段，`t+1` 表示该时段结束后的设备状态。对任何非负功率 `P`，其一个时段对应能量为 `E=P\Delta t`。定义

$$
[x]^+=\max(x,0),\qquad
\operatorname{clip}(x,l,u)=\min\{\max(x,l),u\}.
$$

主要符号：

| 符号 | 含义 |
|---|---|
| `P_{i,t}^{\rm PV},P_{i,t}^{\rm WT},P_{i,t}^{\rm RES}` | 光伏、风电、可再生能源总出力 |
| `L_{i,t}^{\rm base},L_{i,t}^{\rm flex},L_{i,t}` | 基础负荷、柔性设备负荷、实际总负荷 |
| `P_{i,t}^{\rm ch},P_{i,t}^{\rm dis}` | 储能**实际执行**的充电、放电功率（均非负） |
| `P_{i,t}^{\rm p2p,b},P_{i,t}^{\rm p2p,s}` | P2P 实际购电、售电功率（均非负） |
| `P_{i,t}^{\rm grid,b},P_{i,t}^{\rm grid,s}` | 公共电网实际购电、售电功率（均非负） |
| `P_{i,t}^{\rm unmet},P_{i,t}^{\rm curt}` | 供电缺额、被结算为弃电的功率 |
| `\pi_{i,t}^{\rm b},\pi_{i,t}^{\rm s}` | 微网外部电网购电价、售电价 |
| `s_{i,t}` | 微网 `i` 的储能 SOC |

**符号约定**：`\rm b` 表示买入，`\rm s` 表示卖出；在功率平衡式中充电属于消耗、放电属于供给。储能和柔性设备存在“动作请求—约束映射—实际执行”的区别；不能把请求功率直接当成状态方程的实际功率。

## 2. 可再生能源与负荷模型

### 2.1 光伏出力

当前光伏模型基于辐照度的线性容量因子，额定辐照度为 `G_{\rm ref}=1000\ {\rm W/m^2}`：

$$
P_{i,t}^{\rm PV}
=
P_i^{\rm PV,r}
\operatorname{clip}
\left(\frac{G_{i,t}}{G_{\rm ref}},\,0,\,1\right).
$$

其中 `G_{i,t}` 为对应时刻辐照度，`P_i^{\rm PV,r}` 为装机额定功率。MG1 和 MG3 有光伏；MG2 的光伏额定功率为零。当前公式未单独建模逆变器动态、组件温度修正或有功无功耦合。

### 2.2 风电出力

采用切入、额定和切出风速控制的分段功率曲线：

$$
P_{i,t}^{\rm WT}
=
P_i^{\rm WT,r}
\begin{cases}
0, & v_{i,t}<v_{\rm in},\\[3pt]
\dfrac{v_{i,t}^{3}-v_{\rm in}^{3}}
      {v_{\rm r}^{3}-v_{\rm in}^{3}},
& v_{\rm in}\le v_{i,t}<v_{\rm r},\\[9pt]
1, & v_{\rm r}\le v_{i,t}<v_{\rm out},\\[3pt]
0, & v_{i,t}\ge v_{\rm out}.
\end{cases}
$$

$$
v_{\rm in}=3\ {\rm m/s},\qquad
v_{\rm r}=12\ {\rm m/s},\qquad
v_{\rm out}=25\ {\rm m/s}.
$$

MG2 配置 900 kW 风电；MG1 和 MG3 风电额定功率为零。当前时段总可再生能源功率为

$$
P_{i,t}^{\rm RES}=P_{i,t}^{\rm PV}+P_{i,t}^{\rm WT}.
$$

### 2.3 基础负荷与柔性负荷

每个微网的实际电负荷由不可控基础负荷和各自柔性设备用电共同组成：

$$
L_{i,t}=L_{i,t}^{\rm base}+L_{i,t}^{\rm flex},\qquad
L_{i,t}^{\rm flex}\ge 0.
$$

MG1 的柔性负荷为磨机用电，MG2 为 HVAC 的**电功率绝对值**，MG3 为 EV 充电功率。基础负荷不包含上述柔性设备；电网和 P2P 结算均以**实际总负荷** `L_{i,t}` 而非仅基础负荷为供需依据。

## 3. 公共储能（ESS）物理模型

三个微网使用相同形式的储能模型，但容量和功率额定值不同。`P_{i,t}^{\rm ch}` 与 `P_{i,t}^{\rm dis}` 均表示实际执行功率，充放电效率分别为 `\eta_i^{\rm ch}=\eta_i^{\rm dis}=0.95`。

### 3.1 SOC 动态与约束

$$
s_{i,t+1}
=
s_{i,t}
+\frac{\eta_i^{\rm ch}P_{i,t}^{\rm ch}\Delta t}{E_i^{\rm ESS}}
-\frac{P_{i,t}^{\rm dis}\Delta t}
{\eta_i^{\rm dis}E_i^{\rm ESS}}.
$$

$$
s_i^{\min}\le s_{i,t}\le s_i^{\max},
\qquad
s_i^{\min}=0.10,\quad s_i^{\max}=0.90,\quad s_{i,0}=0.50.
$$

时段开始时的瞬时可充、可放电功率分别为：

$$
\overline P_{i,t}^{\rm ch}
=
\min\left\{
P_i^{\rm ch,r},
\frac{(s_i^{\max}-s_{i,t})E_i^{\rm ESS}}
{\eta_i^{\rm ch}\Delta t}
\right\},
$$

$$
\overline P_{i,t}^{\rm dis}
=
\min\left\{
P_i^{\rm dis,r},
\frac{(s_{i,t}-s_i^{\min})E_i^{\rm ESS}
\eta_i^{\rm dis}}{\Delta t}
\right\}.
$$

$$
0\le P_{i,t}^{\rm ch}\le\overline P_{i,t}^{\rm ch},
\qquad
0\le P_{i,t}^{\rm dis}\le\overline P_{i,t}^{\rm dis},
\qquad
P_{i,t}^{\rm ch}P_{i,t}^{\rm dis}=0.
$$

这里的上限只约束**当前时段**的可行充放电功率；一天结束时回到初始 SOC 的要求并没有被提前硬性投影到当前动作，相关偏差在终端作为成本处理。

### 3.2 储能使用与吞吐量成本

储能放电可用于本地负荷，满足本地优先规则后才形成向其他微网的储能售电要约；储能充电可由本地可再生能源、P2P 买入或剩余电网进口能力供给。最终状态更新使用实际放电量和实际获得的充电量：

$$
P_{i,t}^{\rm dis}
=
P_{i,t}^{\rm ESS\to local}
+
P_{i,t}^{\rm ESS\to p2p},
$$

$$
P_{i,t}^{\rm ch}
=
P_{i,t}^{\rm RES\to ESS}
+
P_{i,t}^{\rm p2p\to ESS}
+
P_{i,t}^{\rm grid\to ESS}.
$$

线性电池退化成本为：

$$
C_{i,t}^{\rm deg}
=
c_i^{\rm deg}
\left(P_{i,t}^{\rm ch}+P_{i,t}^{\rm dis}\right)\Delta t,
\qquad
c_i^{\rm deg}=0.01\ {\rm /kWh}.
$$

未成交的储能售电**意向量**不计入实际放电；若外部电网剩余进口能力不足，也允许实际充电小于请求充电。

## 4. 三类异构柔性设备

### 4.1 MG1：工业磨机与料仓

令 `u_t^{\rm mill}\in[0,1]` 为磨机实际运行比例（占空比），则

$$
P_t^{\rm mill}=P^{\rm mill,r}u_t^{\rm mill},
\qquad P^{\rm mill,r}=28.5\ {\rm kW},
$$

$$
q_t^{\rm prod}
=
q^{\rm prod,r}u_t^{\rm mill},
\qquad
q^{\rm prod,r}=1.186\ {\rm t/h}.
$$

料仓库存 `S_t` 的状态转移为

$$
S_{t+1}
=
S_t+
\left(q^{\rm prod,r}u_t^{\rm mill}
-q^{\rm down}\right)\Delta t,
\qquad q^{\rm down}=0.975\ {\rm t/h}.
$$

$$
S^{\min}\le S_t\le S^{\max},
\quad
S^{\min}=3.51\ {\rm t},
\quad S^{\max}=15.80\ {\rm t},
\quad S_0=10.53\ {\rm t}.
$$

由下一时刻料仓容量推导本时段运行比例的瞬时可行区间：

$$
u_t^{\min}
=
\max\left\{
0,\frac{S^{\min}-S_t+q^{\rm down}\Delta t}
{q^{\rm prod,r}\Delta t}
\right\},
$$

$$
u_t^{\max}
=
\min\left\{
1,\frac{S^{\max}-S_t+q^{\rm down}\Delta t}
{q^{\rm prod,r}\Delta t}
\right\}.
$$

磨机动作先投影到上述区间。如果发生供电缺额，其**实际获得的柔性负荷供电比例**记为 `f_t\in[0,1]`，则实际生产和库存更新使用 `u_t^{\rm mill,served}=f_tu_t^{\rm mill,planned}`。因此“计划磨机用电”与“实际完成的生产”在供电不足时不应混为一谈；库存越界异常还会被标记，库存数值在设备提交时截断到配置边界。

### 4.2 MG2：商业 HVAC 的双热容双热阻模型

定义 `T_t^{\rm a}` 为室内空气温度、`T_t^{\rm m}` 为建筑热质量温度、`T_t^{\rm out}` 为室外温度。`R_{\rm oa}` 是室外—空气热阻，`R_{\rm am}` 是空气—热质量热阻；`C_{\rm a},C_{\rm m}` 为对应热容。

室内空气的离散状态方程：

$$
T_{t+1}^{\rm a}
=
T_t^{\rm a}
+
\frac{\Delta t}{C_{\rm a}}
\left[
\frac{T_t^{\rm out}-T_t^{\rm a}}{R_{\rm oa}}
+
\frac{T_t^{\rm m}-T_t^{\rm a}}{R_{\rm am}}
+
Q_t^{\rm int}
+
Q_t^{\rm sol,a}
+
Q_t^{\rm HVAC}
\right].
$$

建筑热质量的离散状态方程：

$$
T_{t+1}^{\rm m}
=
T_t^{\rm m}
+
\frac{\Delta t}{C_{\rm m}}
\left[
\frac{T_t^{\rm a}-T_t^{\rm m}}{R_{\rm am}}
+
Q_t^{\rm sol,m}
\right].
$$

模型参数：

$$
R_{\rm oa}=0.22,\quad R_{\rm am}=0.08\quad
({}^{\circ}{\rm C/kW}),\qquad
C_{\rm a}=15,\quad C_{\rm m}=250
\quad({\rm kWh}/{}^{\circ}{\rm C}).
$$

内部得热和太阳得热按当前配置计算：

$$
Q_t^{\rm sol}=0.025\,G_t,\qquad
Q_t^{\rm sol,a}=0.20\,Q_t^{\rm sol},\qquad
Q_t^{\rm sol,m}=0.80\,Q_t^{\rm sol},
$$

$$
Q_t^{\rm int}
=
\begin{cases}
0.50\,L_t^{\rm base}, & \text{工作日占用时段},\\
0.20\,L_t^{\rm base}, & \text{其余时段}.
\end{cases}
$$

上述太阳得热系数是当前环境的简化模型参数，不应将其理解为完整的建筑采光或热辐射机理。

采用**带符号的 HVAC 电功率** `p_t^{\rm HVAC}\in[-35,35]`：正值代表制热，负值代表制冷。它进入电负荷的量为非负功率：

$$
L_t^{\rm flex}=\left|p_t^{\rm HVAC}\right|,
\qquad
Q_t^{\rm HVAC}
=
\begin{cases}
3.3\,p_t^{\rm HVAC}, & p_t^{\rm HVAC}\ge0,\\
3.2\,p_t^{\rm HVAC}, & p_t^{\rm HVAC}<0.
\end{cases}
$$

工作日 07:00—18:59 对应占用时段，空气舒适温度区间为 `[21,25]\,{}^\circ{\rm C}`；其他时段为 `[16,30]\,{}^\circ{\rm C}`。环境根据**动作执行后 `t+1` 时刻适用的舒适区间**和当前功率范围计算单步 HVAC 动作区间，再截断智能体请求。当功率范围不足以满足舒适要求时，会记录舒适度不可行情况，而不是假定舒适约束必然成立。

若实际供电比例为 `f_t`，设备用 `f_t p_t^{\rm HVAC,planned}` 重新计算实际供热/制冷量及下一时刻温度；不能用未获得供电的计划值直接更新热状态。

### 4.3 MG3：聚合 EV 充电

EV 车队仅允许充电，不允许车辆向外放电（V2G）。可充电时段为 `t=0,\ldots,6`，07:00 离站后不能再安排充电。设 `s_t^{\rm EV}` 为车队聚合 SOC，`P_t^{\rm EV}\ge0` 为实际充电功率，则

$$
s_{t+1}^{\rm EV}
=
\min\left\{
s_{\max}^{\rm EV},
s_t^{\rm EV}
+\frac{\eta^{\rm EV}P_t^{\rm EV}\Delta t}{E^{\rm EV}}
\right\}.
$$

$$
E^{\rm EV}=780\ {\rm kWh},\quad
\eta^{\rm EV}=0.93,\quad
s_0^{\rm EV}=0.50,\quad
s_{\rm target}^{\rm EV}=s_{\max}^{\rm EV}=0.80.
$$

充电时间窗和 SOC 共同决定瞬时充电上限：

$$
0\le P_t^{\rm EV}\le
\begin{cases}
\displaystyle\min\left\{
93.6,\,
\frac{(s_{\max}^{\rm EV}-s_t^{\rm EV})E^{\rm EV}}
{\eta^{\rm EV}\Delta t}
\right\}, & 0\le t<7,\\[10pt]
0, & 7\le t<24.
\end{cases}
$$

离站 SOC 目标不是提前强行保证满足的硬约束；目标未达到时，环境根据终端剩余 SOC 缺口计入经济惩罚。由于离站后无充电，日末 SOC 与离站后的 SOC 相同。

## 5. 本地调度与 P2P 交易

### 5.1 从实际负荷确定本地供需

完成柔性设备动作的瞬时可行域映射后，环境将其负荷加入基础负荷，再处理储能和能量交易。可再生能源先满足本地负荷：

$$
P_{i,t}^{\rm RES\to load}
=\min\{P_{i,t}^{\rm RES},L_{i,t}\},
$$

$$
D_{i,t}^{\rm local}
=[L_{i,t}-P_{i,t}^{\rm RES}]^+,\qquad
S_{i,t}^{\rm RES}
=[P_{i,t}^{\rm RES}-L_{i,t}]^+.
$$

对于储能请求放电 `P_{i,t}^{\rm dis,req}`，本地优先供电量为

$$
P_{i,t}^{\rm ESS\to local}
=\min\{P_{i,t}^{\rm dis,req},D_{i,t}^{\rm local}\}.
$$

储能放电请求尚余的部分才可能形成向其他微网的售电要约，且受到设备剩余可放电功率、对方需求、双边容量与价格可交易条件限制。**售电要约不等于最终交易量**：

$$
0\le P_{i,t}^{\rm ESS\to p2p}
\le P_{i,t}^{\rm ESS,offer}.
$$

当储能正在请求充电，或本地负荷缺额尚未由拟定的本地储能放电覆盖时，不形成该储能的 P2P 售电要约。未成交部分不转化为实际储能放电。

本地多余可再生能源优先满足本地储能充电请求，其余构成可再生能源 P2P 售电要约：

$$
P_{i,t}^{\rm RES\to ESS}
=\min\{P_{i,t}^{\rm ch,req},S_{i,t}^{\rm RES}\},
$$

$$
P_{i,t}^{\rm RES,offer}
=[S_{i,t}^{\rm RES}-P_{i,t}^{\rm RES\to ESS}]^+.
$$

相应地，P2P 买方需求分为**剩余负荷**与**剩余储能充电**两个用途，而不把两者混为一个无优先级的总量。

### 5.2 P2P 撮合规则

能源池按照以下先后顺序依次撮合，每一阶段使用上一阶段之后仍可用的供给、需求和双边容量：

1. 可再生能源 `\to` 其他微网负荷；
2. 储能 `\to` 其他微网负荷；
3. 可再生能源 `\to` 其他微网储能充电；
4. 储能 `\to` 其他微网储能充电。

同一阶段内采用按买方剩余需求比例提出分配、再依据买方需求与双边剩余容量缩放的迭代撮合过程；并非每条交易独立先到先得，也不是对全局 P2P 交易量求解一个单独的线性规划。

设 `F_{ij,t}^{(k)}\ge0` 为第 `k` 阶段由微网 `i` 向 `j` 实际成交的功率，且 `F_{ii,t}^{(k)}=0`。同一微网对的双向成交和全部阶段共用容量：

$$
\sum_{k=1}^{4}\left(
F_{ij,t}^{(k)}+F_{ji,t}^{(k)}
\right)\le75\ {\rm kW},
\qquad i<j.
$$

对每个卖方与买方，还必须分别满足其剩余有效供给与有效需求。只有当买方电网购电价不低于卖方电网售电价时，该交易方向才进入候选撮合：

$$
\pi_{j,t}^{\rm b}\ge\pi_{i,t}^{\rm s}.
$$

总成交功率定义为

$$
P_{i,t}^{\rm p2p,s}
=\sum_{j\ne i}\sum_{k=1}^{4}F_{ij,t}^{(k)},\qquad
P_{i,t}^{\rm p2p,b}
=\sum_{j\ne i}\sum_{k=1}^{4}F_{ji,t}^{(k)}.
$$

当前配置 `loss\_rate=0`，因此整个联盟内的 P2P 功率守恒：

$$
\sum_{i\in\mathcal M}P_{i,t}^{\rm p2p,s}
=
\sum_{i\in\mathcal M}P_{i,t}^{\rm p2p,b}.
$$

**重要边界**：当前 P2P 模型是容量约束下的合作式能量撮合，不是完整线路潮流、节点电压、线路阻抗与损耗联合求解，也不是基于买卖双方自主报价的竞价市场。

### 5.3 P2P 价格与内部收益转移

每条实际成交的 `i\to j` 交易采用卖方外部售电价和买方外部购电价的中间价：

$$
\pi_{ij,t}^{\rm p2p}
=\frac{\pi_{i,t}^{\rm s}+\pi_{j,t}^{\rm b}}{2}.
$$

$$
M_{ij,t}^{\rm p2p}
=
\left(\sum_{k=1}^{4}F_{ij,t}^{(k)}\right)
\Delta t\,\pi_{ij,t}^{\rm p2p}.
$$

每笔交易由买方支付、卖方收取，因此内部交易付款满足

$$
\sum_{i\in\mathcal M}M_{i,t}^{\rm p2p,net}=0.
$$

单个微网的账面收益包含其 P2P 净收款，但内部转移在联盟求和时抵消。**P2P 中间价影响各微网收益分配，不作为额外联盟净收益加入团队奖励。**

## 6. 外部电网、功率平衡与电价

### 6.1 电网结算顺序

P2P 撮合结束后，公共电网首先供应尚未满足的实际负荷，剩余进口容量再用于满足尚未获得充电电源的储能充电请求。若仍有正向富余功率，则在出口功率上限内售电，超过部分记作弃电；若负向缺额超过进口功率上限，则差额记作未满足负荷。

设

$$
N_{i,t}
=
P_{i,t}^{\rm RES}+P_{i,t}^{\rm dis}
-L_{i,t}-P_{i,t}^{\rm ch}
+P_{i,t}^{\rm p2p,b}
-P_{i,t}^{\rm p2p,s}.
$$

实际购电、售电、缺供和弃电由下面的分段结算给出（各微网进口、出口限额均为 200 kW）：

$$
P_{i,t}^{\rm grid,s}
=\min\{[N_{i,t}]^+,\,200\},
\qquad
P_{i,t}^{\rm curt}
=[N_{i,t}-200]^+,
$$

$$
P_{i,t}^{\rm grid,b}
=\min\{[-N_{i,t}]^+,\,200\},
\qquad
P_{i,t}^{\rm unmet}
=[-N_{i,t}-200]^+.
$$

以上公式中的 `P_{i,t}^{\rm ch}` 已是根据本地可再生能源、P2P 和电网**实际可供给能力**确定的充电功率，而非未经电力平衡处理的请求值。当前环境不允许把从公共电网买入的电量直接转售给 P2P 能源池。

### 6.2 每个微网的最终功率平衡

$$
\boxed{
\begin{aligned}
&P_{i,t}^{\rm RES}
+P_{i,t}^{\rm dis}
+P_{i,t}^{\rm p2p,b}
+P_{i,t}^{\rm grid,b}
+P_{i,t}^{\rm unmet}\\
&\quad=
L_{i,t}^{\rm base}
+L_{i,t}^{\rm flex}
+P_{i,t}^{\rm ch}
+P_{i,t}^{\rm p2p,s}
+P_{i,t}^{\rm grid,s}
+P_{i,t}^{\rm curt}.
\end{aligned}}
$$

这里的 `P^{\rm unmet}` 是用于描述供电缺口的平衡量，**不是实际发电或购电来源**；`P^{\rm curt}` 也不是被负荷消耗的电量。若发生缺供，柔性设备用电及状态更新还会考虑实际供电比例，不应由平衡式误推“所有计划负荷均得到满足”。

### 6.3 各微网的外部电价

外部电价按微网分别设置；本文以当前配置货币单位/ kWh 表示：

**MG1：固定购售电价**

$$
\pi_{1,t}^{\rm b}=0.60,\qquad
\pi_{1,t}^{\rm s}=0.18.
$$

**MG2：基于时变参考价 `\pi_t^{\rm ref}` 的价格**

$$
\pi_{2,t}^{\rm b}
=
\operatorname{clip}
\left(0.78\pi_t^{\rm ref}+0.18,\ 0.38,\ 1.08\right),
\qquad
\pi_{2,t}^{\rm s}=0.38\,\pi_{2,t}^{\rm b}.
$$

**MG3：峰谷分时购电价、固定售电价**

$$
\pi_{3,t}^{\rm b}
=
\begin{cases}
0.44, & t\in\{0,1,2,3,4,5\},\\
1.35, & t\in\{16,17,18,19,20,21\},\\
0.82, & \text{其他时段},
\end{cases}
\qquad
\pi_{3,t}^{\rm s}=0.16.
$$

## 7. 运行成本、终端偏差与优化目标

### 7.1 时段经济核算

每个微网的公共电网购电成本、售电收益、缺供成本、弃电成本分别为

$$
C_{i,t}^{\rm buy}
=\pi_{i,t}^{\rm b}P_{i,t}^{\rm grid,b}\Delta t,
\qquad
R_{i,t}^{\rm sell}
=\pi_{i,t}^{\rm s}P_{i,t}^{\rm grid,s}\Delta t,
$$

$$
C_{i,t}^{\rm unmet}
=20\,P_{i,t}^{\rm unmet}\Delta t,
\qquad
C_{i,t}^{\rm curt}
=0.02\,P_{i,t}^{\rm curt}\Delta t.
$$

定义不含终端惩罚的时段物理净收益：

$$
\Pi_{i,t}^{\rm phys,pre}
=
R_{i,t}^{\rm sell}
-C_{i,t}^{\rm buy}
-C_{i,t}^{\rm deg}
-C_{i,t}^{\rm unmet}
-C_{i,t}^{\rm curt}.
$$

P2P 内部净收款只进入单个微网的`账面收益`：

$$
\Pi_{i,t}^{\rm individual}
=\Pi_{i,t}^{\rm phys}+M_{i,t}^{\rm p2p,net}.
$$

### 7.2 日末软约束：ESS、磨机、HVAC 和 EV

终端偏差仅在一日最后一步完成设备状态更新后计算，惩罚系数统一为 `\lambda_{\rm term}=1.5`（货币单位/等效 kWh）。设 `[x]^+=\max(x,0)`。

**储能日末 SOC 回归偏差：**

$$
E_i^{\rm ESS,dev}
=
E_i^{\rm ESS}
\left[
|s_{i,T}-s_{i,0}|-0.03
\right]^+.
$$

**MG1 磨机料仓日末回归偏差：**

$$
E_1^{\rm mill,dev}
=
\frac{P^{\rm mill,r}}{q^{\rm prod,r}}
\left[
|S_T-S_0|-0.20
\right]^+.
$$

上式将超出料仓终端容差的吨数按额定单位产量电耗转换为等效 kWh。

**MG2 HVAC 日末温度回归偏差：**

$$
\delta_{\rm a}
=
[|T_T^{\rm a}-T_0^{\rm a}|-1.0]^+,
\qquad
\delta_{\rm m}
=
[|T_T^{\rm m}-T_0^{\rm m}|-0.3]^+.
$$

$$
E_2^{\rm HVAC,dev}
=
\frac{C_{\rm a}\delta_{\rm a}}{\mathrm{COP}_{\rm a}}
+
\frac{C_{\rm m}\delta_{\rm m}}{\mathrm{COP}_{\rm m}},
$$

其中对 `x\in\{\rm a,m\}`，若对应终端温度高于初始温度则选 `\mathrm{COP}_x=3.2`（制冷），否则选 `\mathrm{COP}_x=3.3`（制热）。这是代码实际采用的**等效电量惩罚换算规则**，而不是额外模拟一次终端 HVAC 恢复过程。

**MG3 EV 目标 SOC 缺口：**

$$
E_3^{\rm EV,dev}
=
E^{\rm EV}[s_{\rm target}^{\rm EV}-s_T^{\rm EV}]^+.
$$

所有不适用于某一微网的特有设备偏差均取零，定义

$$
E_i^{\rm term,dev}
=
E_i^{\rm ESS,dev}
+
E_i^{\rm mill,dev}
+
E_i^{\rm HVAC,dev}
+
E_i^{\rm EV,dev},
$$

$$
C_i^{\rm term}=1.5\,E_i^{\rm term,dev}.
$$

用指示函数 `\mathbf1_{\{t=T-1\}}` 表示终端时段，得到含终端惩罚的时段物理净收益：

$$
\Pi_{i,t}^{\rm phys}
=
\Pi_{i,t}^{\rm phys,pre}
-
\mathbf1_{\{t=T-1\}}C_i^{\rm term}.
$$

**注意**：期末 SOC、料仓库存、建筑温度、EV 目标是软惩罚，不等价于逐时提前强制达到终端目标。当前 DR 环境没有额外的中间时段奖励塑形项。

### 7.3 联盟经济目标

从实际经济核算角度，联盟一日联合总成本可定义为

$$
C_{\rm team}^{\rm day}
=
\sum_{t=0}^{T-1}\sum_{i\in\mathcal M}
\left(
C_{i,t}^{\rm buy}
+C_{i,t}^{\rm deg}
+C_{i,t}^{\rm unmet}
+C_{i,t}^{\rm curt}
-R_{i,t}^{\rm sell}
\right)
+
\sum_{i\in\mathcal M}C_i^{\rm term}.
$$

相应的经济目标是

$$
\min_{\text{各时段可执行调度}}\;
\mathbb E\!\left[C_{\rm team}^{\rm day}\right],
$$

其中期望表示跨不同时段条件和运行日的总体运行表现。**该式是系统经济目标的表达，不意味着当前代码每步都求解一个精确的全日数学规划。** 当前强化学习训练通过下一节的折扣奖励间接学习调度策略。

现有奖励函数中**没有单列峰值削减、峰谷差、碳排放或 Shapley 分摊的奖励项**。P2P 通过改变实际能源分配和外部购售电结算间接影响联盟总成本。

## 8. 多智能体强化学习建模

### 8.1 智能体及集中训练、分散执行

$$
\mathcal A=\{\mathrm{MG1},\mathrm{MG2},\mathrm{MG3}\}.
$$

三个微网分别对应一个 Actor，根据各自观测给出调度动作。训练采用 CTDE（集中训练、分散执行）：Critic 在训练时可以利用全局状态和联合动作；Actor 执行时使用分配给对应微网的观测。

当前普通 MATD3 基准并非“一个共享 Critic + 三个 Actor”，而是**三个独立 Actor、三个分别对应各微网的集中式双 Q Critic**，三个 Critic 所学习的即时团队奖励相同。Transformer 对照算法主要改变 Actor 的信息编码结构，不改变这里说明的物理环境目标。

### 8.2 原始动作与有效动作

每个 Actor 在时刻 `t` 输出两个归一化原始动作：

$$
\mathbf a_{i,t}^{\rm raw}
=
\begin{bmatrix}
a_{i,t}^{\rm ESS}\\
a_{i,t}^{\rm DR}
\end{bmatrix}
\in[-1,1]^2.
$$

储能原始动作正值请求放电、负值请求充电。DR 动作含义随设备而不同：

$$
u_t^{\rm mill,req}
=\frac{a_{1,t}^{\rm DR}+1}{2},
\qquad
p_t^{\rm HVAC,req}
=35\,a_{2,t}^{\rm DR},
\qquad
P_t^{\rm EV,req}
=93.6\,\frac{a_{3,t}^{\rm DR}+1}{2}.
$$

请求动作经过柔性设备瞬时可行域和储能本地优先调度映射后，形成训练使用的三个有效动作通道：

$$
\mathbf a_{i,t}^{\rm eff}
=
\begin{bmatrix}
\left(P_{i,t}^{\rm ESS\to local}
-P_{i,t}^{\rm ch,req}\right)/P_i^{\rm ESS,r}\\[2pt]
P_{i,t}^{\rm ESS,offer}/P_i^{\rm ESS,r}\\[2pt]
a_{i,t}^{\rm DR,eff}
\end{bmatrix},
\qquad
P_i^{\rm ESS,r}
=\max(P_i^{\rm ch,r},P_i^{\rm dis,r}).
$$

其中 `a_{i,t}^{\rm DR,eff}` 是柔性设备约束映射后的归一化值；第二通道是储能 **P2P 售电要约**，并非撮合完成后的实际成交功率。实际成交和充电可获得量要由环境 P2P/电网流程进一步确定。

### 8.3 观测与全局状态

每个微网的 Actor 获得本地实时供需、储能 SOC、设备状态、价格、可行操作区间、跨微网协同摘要及未来时段预测等观测。全局状态将多个微网的相应信息用于集中训练。终端目标所需的剩余时间和偏差压力会作为信息提供，但**提供信息不等于环境替智能体执行未来动作**。

当前环境配置的基础预测长度为 24 h，超出当日部分可屏蔽；普通分层 Transformer Actor 的配置中采用 8 h 的预测 token 窗口。**环境可提供的预测范围与某个 Actor 实际采用的窗口并不是同一概念。**

### 8.4 共享团队奖励

代码计算每个时段的联盟物理净收益之和，并对每个 Actor 返回同一个团队奖励：

$$
\Pi_t^{\rm team}
=
\sum_{i\in\mathcal M}\Pi_{i,t}^{\rm phys},
\qquad
\boxed{
r_{1,t}=r_{2,t}=r_{3,t}
=\frac{\Pi_t^{\rm team}}{100}
}.
$$

P2P 内部付款是零和转移，不额外计入 `\Pi_t^{\rm team}`。终端惩罚通过最后一步 `\Pi_{i,T-1}^{\rm phys}` 进入共享奖励。

从强化学习训练角度，策略目标可以写为

$$
\max_{\boldsymbol\pi}
J(\boldsymbol\pi)
=
\mathbb E_{\boldsymbol\pi}
\left[
\sum_{t=0}^{T-1}\gamma^t r_t
\right],
\qquad
\gamma=0.99.
$$

**区别**：最小化不折扣的一日真实经济成本，与最大化 `\gamma=0.99` 的折扣回报并非严格同一个数学目标。展示实际运行表现、与 MILP 进行经济性对照时，应比较相同场景下的真实结算成本及约束表现，而不是直接比较折扣回报值。

## 9. 环境一次调度的实际执行流程

以下流程串联前述公式，特别用于阅读系统能量分配逻辑：

1. 读取当前时段基础负荷、风光出力、外部电价和设备状态，接收三个 Actor 的原始动作。
2. 先对磨机、HVAC、EV 动作进行当前时段物理可行域映射，得到**计划柔性用电**，形成各微网实际计划总负荷。
3. 结合实际计划负荷计算储能动作映射：可再生能源先满足本地负荷，储能放电本地优先；按剩余可用功率及对方需求形成储能 P2P 售电要约。
4. 计算本地可再生能源对储能的充电供给，形成剩余可再生能源售电要约、其他微网尚未满足的负荷需求和储能充电需求。
5. 按“可再生能源→负荷、储能→负荷、可再生能源→储能、储能→储能”的四阶段顺序执行 P2P 撮合，得到真实成交量。
6. 按真实成交量修正储能实际放电；再由电网优先供应剩余负荷，剩余电网进口容量用于储能充电。确认储能实际充电及缺供、弃电。
7. 用**实际充放电**更新 ESS SOC；若供电不足，柔性设备根据实际供电比例更新磨机生产与库存、HVAC 温度或 EV SOC。
8. 结算电网购售电、储能退化、缺供、弃电及 P2P 账面转移；仅在当日最后一个时段计算终端偏差惩罚。
9. 计算联盟共享奖励，更新跨微网协同历史，生成下一时刻观测并进入下一步。

其中“实际计划总负荷”用于安排供需与撮合；最终缺供出现时，设备状态更新还要考虑真实交付比例。研究环境的执行先后时应将这两个层次分开。

## 10. 建模边界与阅读注意事项

- 本系统使用 1 h 离散功率平衡和 P2P 成交容量模型；**未建立完整 AC 潮流、节点电压、线路阻抗及输电损耗优化**。配置中的 P2P 损耗为零不等于现实微网线路无损耗。
- 当前风光功率曲线、线性储能吞吐量退化成本、2R2C 热模型和聚合 EV 模型，均是研究环境采用的简化物理表示；不可据此宣称涵盖对应设备的所有物理动态。
- 柔性负荷虽然存在当前时段可行域映射，但供电短缺可能使设备实际状态偏离计划，并可能被标记为不可行；不应写成“任意情况下所有设备约束均严格满足”。
- 期末储能、库存、温度及 EV 目标以**软惩罚**而不是终端硬性强制可达的方式处理。EV 在 07:00 离站，日末目标惩罚根据离站后保持的 SOC 计算。
- 当前 P2P 为确定性合作能源池及中间价结算，不属于纳什议价、策略竞价或 Shapley 收益分配模型。
- 联盟目标侧重真实运行经济性；当前奖励函数**没有明确的独立削峰目标项**，不能仅凭该公式声称已实现直接的峰值最小化。

---

**基准来源**：本说明的物理设定、能量分配及奖励解释依据 `代号：噜噜` 分支上述提交中的 `configs/cooperative.yaml`、`core/profiles.py`、`components/battery.py`、`components/industrial_mill.py`、`components/commercial_hvac.py`、`components/ev_charging.py`、`core/joint_action_projection.py`、`core/single_dispatch.py`、`envs/cooperative_energy_sharing.py`、`core/grid_settlement.py`、`core/tariffs.py`、`core/cost_accounting.py` 与 `envs/multi_microgrid_env.py`。本文提供的是可读的数学建模概括；对迭代撮合及特殊不可行情形，以该提交的具体实现为准。
