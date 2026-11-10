# Training a Task-Specific AI Assistant with Model Training Toolkit

*Instruction fine-tuning, application contracts, structured responses, and evaluation*

## What are we building?

Model Training Toolkit (MTT) supports a workflow for specializing an existing **instruction-following language model** to perform a well-defined task within an application. Instead of training a general-purpose model from scratch, we provide examples of the requests an assistant will receive and the responses the application expects.

This approach is called **supervised fine-tuning (SFT)**. One implementation uses **Low-Rank Adaptation (LoRA)** to train a relatively small set of parameters while retaining the capabilities of the base model.

The output need not be JSON. An assistant could classify messages, extract fields, construct query plans, or generate another application-defined response. What matters is that the application establishes a clear **contract** for the assistant's behavior.

To make the process concrete, we'll build a hypothetical assistant that converts natural-language requests into records for a to-do list.

## 1. Start with the application contract

Imagine a user typing:

> “I started cleaning the garage yesterday. I need to finish by Saturday, but it's not urgent.”

Rather than asking the user to fill out a form, our assistant extracts values for a simple task table:

| Column | Type | Required? | Rule |
|---|---|---|---|
| `title` | Text | Yes | Brief description of the task |
| `description` | Text or `null` | No | Additional details, if provided |
| `due_date` | `YYYY-MM-DD` or `null` | No | Deadline, if specified |
| `start_date` | `YYYY-MM-DD` or `null` | No | When work started, if specified |
| `priority` | `low`, `medium`, `high` | Yes | Defaults to `medium` |
| `status` | `pending`, `in_progress`, `completed` | Yes | Defaults to `pending` |

The application-layer contract tells the assistant to return exactly these fields as a valid JSON object, with no surrounding explanation. Missing optional values must be `null`; dates must use ISO formatting; and enumerated values must come from the allowed lists. **The assistant must not invent facts that were not supplied.**

Relative dates require context. For the following examples, assume the application explicitly supplies **the current date as Tuesday, November 10, 2026**. The assistant can then interpret “yesterday” as November 9 and “Saturday” as November 14.

The expected response to our garage request is:

```json
{
  "title": "Clean garage",
  "description": null,
  "due_date": "2026-11-14",
  "start_date": "2026-11-09",
  "priority": "low",
  "status": "in_progress"
}
```

**The application owns the contract.** The model learns to produce outputs that comply with it, while the application remains responsible for parsing, validating, and deciding whether to create a database record. The assistant should not write directly to the database merely because it produced valid JSON.

## 2. Teach the contract with examples

For SFT, a training example pairs a request with an approved assistant response. Each example teaches an association between the user's language, relevant context, and the required structure.

### Example A: A task without optional details

**Instruction:** “Add buy groceries to my list.”

**Expected assistant response:**

```json
{
  "title": "Buy groceries",
  "description": null,
  "due_date": null,
  "start_date": null,
  "priority": "medium",
  "status": "pending"
}
```

### Example B: An explicit deadline and urgency

**Instruction:** “Finish the quarterly report by November 20. It's urgent.”

**Expected assistant response:**

```json
{
  "title": "Finish quarterly report",
  "description": null,
  "due_date": "2026-11-20",
  "start_date": null,
  "priority": "high",
  "status": "pending"
}
```

### Example C: Completed work

**Instruction:** “I already paid the electric bill. Mark it complete.”

**Expected assistant response:**

```json
{
  "title": "Pay electric bill",
  "description": null,
  "due_date": null,
  "start_date": null,
  "priority": "medium",
  "status": "completed"
}
```

Notice what the model is learning: how “urgent” maps to `high`, how “already paid” maps to `completed`, and how to leave unspecified optional fields empty rather than filling in plausible guesses.

These are **illustrative input/output pairs**, not a claim about the exact on-disk training schema for every MTT version. An actual MTT project packages examples in its supported dataset format, along with its system prompt and configuration.

### How many examples?

A small demonstration can use a handful of cases, but a useful production assistant might need **hundreds, thousands, or tens of thousands** of carefully reviewed examples. The appropriate amount depends on the task, the starting model, and the diversity of requests.

For the to-do assistant, the training corpus should vary vocabulary, word order, missing fields, priorities, date expressions, task status, and conflicting or ambiguous instructions. More examples alone do not guarantee a better model; coverage, consistency, and quality matter.

## 3. Fine-tune the assistant using MTT

A typical MTT workflow is:

1. **Choose an instruction-following base model**, such as a Qwen Instruct model suited to available hardware.
2. **Define the contract and system instructions**: supported fields, formatting rules, allowed values, defaults, and how to handle missing information.
3. **Prepare the supervised examples**: each request and its expected assistant response.
4. **Configure training**: LoRA settings, learning rate, batch behavior, number of epochs, and other parameters.
5. **Train and retain the artifacts**: model adapter, checkpoints, versioned configuration, frozen input data, and training metadata.
6. **Export for inference**: optionally merge the LoRA adapter into the base model and serve a compatible model artifact.

An **epoch** is one pass through the training dataset. Four epochs means four passes through the examples, not four separate datasets.

In an assistant-oriented fine-tuning implementation, **assistant-only loss masking** can focus the training objective on predicting the correct assistant response rather than asking the model to reproduce user instructions. MTT's training configuration and backend determine the specific implementation details.

After training, the model can be served through a supported inference engine—for example, Hugging Face-compatible model files or a converted GGUF model through Ollama. The serving path and generation settings are part of reproducibility and should be recorded with evaluations.

## 4. Evaluation A: Did the model learn the training examples?

The first check is a **training baseline** (also called training recall). Ask the finished model to generate outputs for the same examples used in training, then score them against the approved answers.

Suppose we trained on 5,000 requests:

| Training-baseline metric | Hypothetical result |
|---|---:|
| Examples | 5,000 |
| Correct structured responses | 4,950 |
| Accuracy | 99.0% |

This is useful for identifying patterns the model failed to learn. We can inspect the 50 failures and categorize them—for example, date interpretation, priority classification, or incomplete outputs.

But **high training recall is not proof of generalization**. The model has already encountered these examples during fine-tuning.

## 5. Evaluation B: Can the model handle unseen requests?

To test generalization, we maintain a separate **blind (held-out) evaluation set** that is excluded from training.

A new request might say:

> “Put the expense report on my list for next Wednesday. It isn't important yet.”

Assuming the same supplied current date of November 10, 2026, the expected structured record would have a November 18 due date, `low` priority, and `pending` status.

The assistant has not seen this exact question in training. We are testing whether it learned the underlying rules and can apply them to unfamiliar wording.

| Evaluation | Hypothetical examples | Correct | Accuracy |
|---|---:|---:|---:|
| Training baseline | 5,000 | 4,950 | 99.0% |
| Blind holdout | 500 | 460 | 92.0% |

The gap helps us understand how well the assistant transfers its learned behavior to new inputs. A model could memorize its training data and still struggle with previously unseen instructions.

**Keep holdout data protected.** If a held-out question is later used for training, it can remain useful as a known-issue regression test, but it is no longer an independent test of generalization.

## 6. Decide what constitutes a pass

We need evaluation rules that reflect the application contract, not just text similarity.

These two responses have identical meaning for a two-field illustration, despite their different field ordering:

```json
{"title":"Buy groceries","priority":"high"}
```

```json
{"priority":"high","title":"Buy groceries"}
```

For the real six-field contract, both would be incomplete because required fields are missing. A **contract-aware scorer** should enforce required fields and valid values while accepting semantically equivalent representations that the contract permits.

Useful measurements include:

- **JSON validity:** Is the response parseable?
- **Contract compliance:** Are the required fields, types, and allowed values correct?
- **Exact match:** Does it match the approved output under a defined normalization rule?
- **Semantic correctness:** Are the extracted facts and their relationships correct, allowing only approved representational differences?

For example, `high` instead of `low` is a substantive error. Swapping JSON key order is not. A custom MTT evaluation runner can apply the rules appropriate to the application.

## 7. Maintain a shared, versioned benchmark

As an assistant evolves, we want newer versions to preserve abilities learned earlier. A **shared benchmark** is a growing set of evaluation cases that multiple model versions can run against.

For our to-do application, we might create the following targeted suites:

| Suite | What it tests |
|---|---|
| `basic_tasks` | Titles, descriptions, and defaults |
| `dates` | Explicit and relative deadlines |
| `priorities` | Importance and urgency |
| `statuses` | Pending, started, and completed tasks |
| `ambiguous_requests` | Missing or conflicting details |
| `full_benchmark` | The complete protected evaluation set |

During development, an engineer might run just the date suite. Before releasing a model, the team runs the canonical full benchmark and any required regression checks.

When a new weakness is discovered, a reviewed test can be added to **a new benchmark package version**, rather than silently changing existing test data. A test incorporated into training should be labeled as a regression/retention check, not an unseen holdout.

Every evaluation should record its **benchmark package version, dataset hashes, scorer identity, model identity, system prompt, and generation settings**. The semantic contract's version is a separate concern: adding new test cases changes the benchmark even when the contract itself stays the same.

To compare two models fairly, evaluate both against **the same benchmark package, exact dataset hashes, and compatible scoring configuration**. Otherwise, a higher score may simply reflect an easier or different set of questions.

## 8. Integrate the trained assistant into the application

Once the model passes its required evaluations, the application can use it to translate natural language into structured data:

```text
User request
    ↓
Application supplies the current date and instructions
    ↓
Fine-tuned assistant generates a structured response
    ↓
Application validates against the contract
    ↓
Application creates or updates the task record
```

For example:

**User:** “Call the dentist tomorrow. High priority.”

**Assistant response:**

```json
{
  "title": "Call dentist",
  "description": null,
  "due_date": "2026-11-11",
  "start_date": null,
  "priority": "high",
  "status": "pending"
}
```

The application checks the fields and then decides whether to persist the task. The model supplies the interpretation; the software remains responsible for enforcing the contract and taking action.

## The core idea

MTT helps developers create **task-specific, instruction-following assistants** using a repeatable process:

**Define the application contract → prepare supervised examples → fine-tune a base model → measure training recall → test on held-out questions → deploy and monitor behavior.**

The to-do list is only an illustration. The same approach can support structured document extraction, classification, natural-language query planning, and many other applications where flexible human instructions must become predictable, verifiable outputs.
