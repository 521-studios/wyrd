variable "env" {
  description = "Deployment environment (staging | production)"
  type        = string
  # wyrd-0gou: the WYRD_FF_ALL conditional (main.tf) keys off env == "staging",
  # so a typo'd env (e.g. "Staging" / "prod") would silently deploy with all
  # flags off. Fail loud at plan time instead.
  validation {
    condition     = contains(["staging", "production"], var.env)
    error_message = "env must be \"staging\" or \"production\"."
  }
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-2"
}

variable "lambda_zip_path" {
  description = "Path to the compiled Lambda zip"
  type        = string
  default     = "../function.zip"
}

variable "log_level" {
  description = "Python logging level for the Lambda runtime (DEBUG | INFO | WARNING | ERROR)"
  type        = string
  default     = "INFO"
}

# wyrd-0gou: SPA feature flags. Staging gets WYRD_FF_ALL=true via the env
# conditional in main.tf (every gated option on for validation); production
# defaults all off and turns validated options on one-by-one by listing their
# flag names here (e.g. ["novelty", "culture.welsh", "moods"]). Each becomes a
# WYRD_FF_<NAME>=true Lambda env var.
variable "enabled_feature_flags" {
  description = "SPA feature-flag names to force on for this environment (each → WYRD_FF_<NAME>=true). Ignored where WYRD_FF_ALL is already true (staging)."
  type        = list(string)
  # wyrd-7f22: novelty is validated + shipping on, so surface its slider in
  # production (staging already shows every option via WYRD_FF_ALL). Both envs
  # read this default — the deploy passes only -var="env=..." (see deploy.yml),
  # so a single default applies to staging + production alike.
  default = ["novelty"]
}

# wyrd-0gou: per-environment overrides for option DEFAULT VALUES (not just
# whether an option shows). Keys are option names (e.g. "culture", "count");
# each becomes a WYRD_DEFAULT_<OPTION> Lambda env var the SPA seeds from.
variable "feature_flag_defaults" {
  description = "Per-option default-value overrides (option name → value); each → WYRD_DEFAULT_<OPTION>."
  type        = map(string)
  # wyrd-7f22: default novelty to 0.1 in both prod + staging — a small,
  # always-on amount of lexical surprise. The schema default stays 0.0 (CLI /
  # local / tests bit-stable); this env override seeds the deployed SPA's
  # novelty slider to 0.1. Resolves to WYRD_DEFAULT_NOVELTY=0.1; the SPA
  # coerces the string to the number 0.1 when seeding (featureFlags.coerceToType).
  default = { novelty = "0.1" }
}
