# terraform-reviewer

Review terraform changes in `terraform/` to ensure this app stays in its lane within the **three-layer state stack**:

```
infra (baseline) → apps (this repo) → infra-frontend
```

**This repo's layer: apps.** Its terraform owns app-specific resources only. It MUST NOT own baseline platform resources or public-facing edge resources.

**Default severity:** **P1** for layering violations or forbidden resource types (CloudFront distributions, public ACM certs, public DNS records, foundational shared resources). **P2** for hardcoded values (account IDs, region literals) or output-shape changes that need coordinated `infra-frontend` updates.

For full context, read `infra/CLAUDE.md` and `infra-frontend/CLAUDE.md` in the workspace before reviewing.

### What this app's terraform SHOULD own

- App compute: Lambda function + Function URL, ECS tasks, ALBs.
- App storage: S3 buckets the app reads or writes (including the SPA bucket).
- App IAM: roles, policies, and inline permissions the app's compute consumes.
- Internal origin DNS records the app owns (e.g. `<app>-origin.521studios.com` pointing at an ALB).
- CloudWatch log groups and alarms scoped to the app.

### What this app's terraform MUST NOT own

- **CloudFront distributions** — owned by `infra-frontend`.
- **ACM certificates** for public domains — owned by `infra-frontend` (these must live in `us-east-1` for CloudFront).
- **Public DNS records** (apex, www, custom subdomains the public hits directly) — owned by `infra-frontend`.
- **CloudFront Functions** — owned by `infra-frontend`.
- **Foundational shared resources**: VPCs, subnets, security groups, Aurora clusters, ECS clusters — owned by `infra`.

### What this app's terraform MUST NOT do

- **Read from `infra-frontend` remote state.** Apps deploy *before* infra-frontend, so this is a circular dependency. If a CloudFront-owned value is needed, the value belongs in the app's outputs and infra-frontend should consume it, not the other way around.
- **Embed AWS account IDs as literals** outside of remote-state backend configs. Use `data "aws_caller_identity"` or variables.
- **Reach across into another app's resources.** Apps consume shared values from `infra` via remote state and shared primitives via AWS data sources — they do not poke into peer apps' state.

### What this app's terraform SHOULD do

- **Export the outputs that `infra-frontend` consumes**: `lambda_function_url`, `lambda_function_name`, `s3_bucket_name`, `s3_bucket_arn`, `s3_bucket_regional_domain`. Naming should match what `infra-frontend` already reads — see the `terraform_remote_state` blocks in `infra-frontend/terraform/environments/<env>/main.tf`.
- **Read from `infra` remote state** when consuming shared platform values (VPC IDs, ECS cluster ARN, DB endpoints).
- **Use AWS data sources** (e.g. `data "aws_route53_zone"`) instead of hardcoding values that already exist in the account.
- **Keep environments separate**: `terraform/environments/staging/` and `terraform/environments/production/` are distinct root modules with distinct state.

### Cost discipline

- A new CloudFront distribution + ACM cert costs ~$0.60/month minimum to exist, before any data transfer. Before suggesting "this app should own its own distribution," ask whether a path behavior on an existing distribution would work — and remember that even when the answer is "yes, a new distribution is justified," the distribution still belongs in `infra-frontend`, not here.

### Review approach

1. For each `resource "aws_*"` and `module ".*"` in the diff, ask: does this belong in the app layer, or is it overreach into `infra` or `infra-frontend`?
2. Flag any `terraform_remote_state` block reading from `infra-frontend/<env>/terraform.tfstate` — that's the smoking gun for a layer violation.
3. Flag any `aws_cloudfront_distribution`, `aws_acm_certificate`, `aws_cloudfront_function`, public-facing `aws_route53_record` (anything not under `*-origin.521studios.com` or similar internal aliases), or VPC/subnet resources.
4. Flag hardcoded account IDs, region literals that mismatch the rest of the repo, or duplicated provider blocks.
5. For new outputs, confirm they have a clear consumer in `infra-frontend` (or another known reader) — orphan outputs accumulate over time.
6. For new variables, confirm sensible defaults and that the staging/production root modules both wire them through.

**Note:** It is acceptable to acknowledge a layering violation and defer the fix by creating a beads ticket — but mark it P2 at minimum, not P3. Layer violations create deploy-order coupling that gets harder to untangle the longer it sits.

