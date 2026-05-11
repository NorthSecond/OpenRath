# Engineering Agents Example

Nested engineering workflow:

```text
EngineeringProjectWorkflow
  -> lead
  -> FeatureSquadWorkflow
       -> architect
       -> BackendPairWorkflow
            -> backend_auth
            -> backend_data
       -> frontend
  -> QualityAssuranceWorkflow
       -> tester
```

This example shows how plain Python `Workflow` subclasses compose multiple
`AgentParam` roles and nested workflows while a `Session` carries context from
one step to the next.

Run from the repository root:

```bash
python example/engineering_agents/main.py \
  --goal "Full-stack todo app with auth, DB, React frontend." \
  --workdir .workspace/engineering-agents
```

Requires OpenAI-compatible LLM settings.
