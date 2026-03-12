# Custom Promptfoo Config (Tier 3)

The three-tier config model lets you progressively take control of the eval:

| Tier | When to use | How |
|---|---|---|
| 1 | Default prompt, default scoring | `supereval run <dataset> --model <id>` |
| 2 | Custom prompt, default scoring | Add `--prompt "..."` |
| 3 | Custom prompt + custom scoring/providers | Add `--config promptfoo.yaml` |

Tiers 1 and 2 cover the majority of use cases. Use Tier 3 when you need:
- Custom assert logic (JavaScript, Python, regex, embedding similarity)
- Custom model parameters (temperature, system prompt, max tokens)
- Multiple providers with different configurations
- Promptfoo features not exposed by supereval (e.g. `defaultTest`, `env` overrides)

## How Tier 3 works

You provide a `promptfoo.yaml` with `prompts` and `providers`. supereval **injects the `tests` section automatically** — you never write the tests in the config file. This means your config stays clean and the dataset remains the single source of truth.

```bash
supereval run my-dataset --config promptfoo.yaml
```

supereval will:
1. Load your `promptfoo.yaml`
2. Build the test cases from `datasets/my-dataset/cases.jsonl`
3. Inject them as the `tests` key
4. Run `promptfoo eval` with the merged config

Any existing `tests` key in your config file is overwritten.

## Example: custom model parameters

Control temperature, max tokens, and system prompt via the provider config:

```yaml
# promptfoo.yaml
prompts:
  - "You are an AWS expert. Answer accurately and cite the relevant service name. Question: {{query}}"

providers:
  - id: anthropic:claude-opus-4-6
    config:
      temperature: 0.0
      max_tokens: 512
      system: "You are a helpful AWS support assistant. Be precise and cite sources."
```

## Example: custom scoring for `qa`

Replace the default contains + LLM judge with a stricter regex assert:

```yaml
# promptfoo.yaml
prompts:
  - "Answer concisely: {{query}}"

providers:
  - anthropic:claude-opus-4-6

defaultTest:
  assert:
    - type: regex
      value: "\\d+ (KB|MB|GB|ms|minutes|seconds)"
      # Only useful for numeric-answer datasets — adapt to your domain
```

Note: when using `defaultTest`, it applies to all test cases. supereval's per-case asserts (from `_build_tests()`) are still injected alongside `defaultTest` asserts. They combine — a case must pass all of them.

## Example: JavaScript custom assert

Verify the response contains the answer and is under a word limit:

```yaml
# promptfoo.yaml
prompts:
  - "Answer the following question in one sentence: {{query}}"

providers:
  - anthropic:claude-opus-4-6

defaultTest:
  assert:
    - type: javascript
      value: |
        const words = output.trim().split(/\s+/).length;
        if (words > 30) {
          return { pass: false, score: 0, reason: `Response too long: ${words} words` };
        }
        return { pass: true, score: 1 };
```

## Example: two providers with different prompts

Compare how two models respond to the same dataset with different prompt styles:

```yaml
# promptfoo.yaml
prompts:
  - id: concise
    raw: "Answer concisely in one sentence: {{query}}"
  - id: detailed
    raw: "Provide a thorough answer with examples: {{query}}"

providers:
  - id: anthropic:claude-opus-4-6
    label: claude-concise
    prompts: [concise]
  - id: openai:gpt-4o
    label: gpt-detailed
    prompts: [detailed]
```

## Example: using a custom grading provider

By default Promptfoo uses its hosted grading model for `llm-rubric` asserts. Override it to use your own Bedrock model:

```yaml
# promptfoo.yaml
prompts:
  - "{{instruction}}{% if document %}\n\n---\n{{document}}{% endif %}"

providers:
  - anthropic:claude-sonnet-4-6

defaultTest:
  options:
    provider:
      id: bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0
      config:
        region: us-east-1
```

This ensures all LLM judge calls in your eval go through Bedrock — no external API calls.

## Viewing the generated config

To see exactly what supereval generates before running:

```bash
supereval dataset export my-dataset --output tests-only.yaml
```

This exports just the `tests` section. Paste it into your `promptfoo.yaml` to start from a concrete baseline when building a Tier 3 config.

## Promptfoo assert types reference

| Type | What it checks |
|---|---|
| `contains` | Output contains string (case-sensitive) |
| `icontains` | Output contains string (case-insensitive) |
| `regex` | Output matches regex |
| `not-contains` | Output does not contain string |
| `javascript` | Custom JS function returns `{ pass, score, reason }` |
| `python` | Custom Python function |
| `llm-rubric` | LLM judge evaluates against a rubric string |
| `model-graded-closedqa` | LLM judge answers a yes/no question about the output |
| `similar` | Embedding similarity to reference text |
| `cost` | Token cost is below threshold |
| `latency` | Response latency is below threshold (ms) |

Full documentation: [promptfoo.dev/docs/configuration/expected-outputs](https://www.promptfoo.dev/docs/configuration/expected-outputs/)
