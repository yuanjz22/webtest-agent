# WebTest Agent

WebTest Agent 把自然语言描述的功能测试交给真实浏览器执行。你提供应用地址、操作要求和预期结果，例如「把 Notebook 的数量改成 2，保存后检查总价是否为 CNY 50.00」。可以在本地工作台查看每一步操作，也可以用命令行批量运行场景。

## 整体思路

1. **规划**。Planner 从场景中提取检查点并安排顺序。例如「修改前数量为 1」和「保存后数量为 2」要在两个时刻分别验证。
2. **操作**。Runner 将当前目标和页面观察交给 Actor。Actor 选择一个动作，Playwright 在独立浏览器会话中执行，再采集页面文字、可见控件及截图。
3. **判定**。Assertor 对照操作前后的观察验证检查点，Verifier 校验引用的文字和截图。所有检查点有证据支持时返回 `PASS`，页面证据与要求矛盾时返回 `FAIL`，执行中断或证据不足时返回 `INCONCLUSIVE`。

三个角色共用 `deepseek-flash`。每个场景最多执行 35 个浏览器动作，模型请求和 token 用量单独记录。浏览器操作限制在目标网站的同一来源，当前主要支持主页面及其开放的 Shadow DOM。

## 快速体验

下面的命令在项目根目录的 PowerShell 中执行。需要 Conda 和本机 Google Chrome。

```powershell
conda env create -f environment.yml
conda activate webtest-agent
webtest demo-server
```

保持这个终端运行。它提供待测的购物车网站，默认地址为 `http://127.0.0.1:18080`。在第二个终端激活同一环境并启动工作台。

```powershell
conda activate webtest-agent
webtest serve
```

浏览器打开 `http://127.0.0.1:8000`，选择「本地演示夹具」，保留默认场景并点击「启动测试」。此模式使用固定购物车流程，无需 API 密钥。在初始购物车上运行时，数量保存为 2、总价为 CNY 50.00 即为预期结果。重跑前可在购物车页面点击 **Reset cart**。

若演示端口不可用，执行 `webtest demo-server --port 0`，并把工作台或场景文件中的应用地址改为终端打印的地址。需要改用 Playwright 配套的 Chromium 时，在运行工作台或测试命令的终端执行：

```powershell
python -m playwright install chromium
$env:WEBTEST_BROWSER_CHANNEL = 'chromium'
```

## 运行自己的测试

在项目根目录将 `.env.example` 复制为 `.env`，填入自己的 `MODEL_API_KEY`。模板中的 `MODEL_BASE_URL` 指向 DeepSeek API。重启 `webtest serve` 后选择「真实模型」，填写待测应用的地址和英文场景。模型接收场景、页面文字及控件信息，截图和浏览器轨迹保存在本地。

命令行和工作台共用同一套执行逻辑。以下三条命令可按需单独运行。

```powershell
# 对正在运行的演示站执行固定流程
webtest run examples/local-demo.json --demo

# 用真实模型运行场景。可在 JSONL 中每行写一个场景
webtest run examples/scenarios.template.jsonl

# 运行项目测试
python -m pytest -q
```

场景包含唯一编号 `id`、英文操作与验收要求 `description`、目标地址 `base_url` 和可选动作上限 `max_steps`。默认策略为 `checkpoint`。启动新的测试时，工作台和命令行都会把运行记录写入 `artifacts/`。在工作台选中任务即可查看判定依据、截图和执行事件，也可用 `python -m playwright show-trace <trace.zip路径>` 回放轨迹。

## 目录结构

```text
webtest-agent/
├── .github/workflows/tests.yml    # CI 在 Chromium 上运行 pytest
├── examples/                      # 单场景 JSON 和批量 JSONL 示例
├── src/webtest_agent/
│   ├── cli.py、__main__.py         # 命令行入口
│   ├── api.py、static/             # FastAPI 工作台接口及 HTML、CSS、JavaScript
│   ├── service.py、runner.py       # 组装角色、调度动作、控制步数
│   ├── planner.py、actor.py、assertor.py
│   ├── browser.py、verifier.py     # 浏览器操作、观察和证据检查
│   ├── llm.py、models.py           # 模型客户端和数据结构
│   ├── storage.py、evaluation.py   # 运行记录、材料导出和离线评测
│   └── demo.py、demo_planner.py    # 本地购物车及固定演示流程
├── tests/                         # 核心逻辑、存储和真实浏览器测试
├── .env.example                   # 模型配置模板
├── environment.yml                # Conda 环境
├── pyproject.toml                 # 包元数据、依赖及 webtest 命令
├── requirements-lock.txt          # 依赖版本清单
└── README.md
```

