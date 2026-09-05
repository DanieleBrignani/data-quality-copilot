"""High-level AI tasks: the only place the application asks Anthropic for anything.

Each task follows the same four steps:

1. reduce the dataset to metadata (:mod:`dqcopilot.ai.payload`);
2. ask for a schema-constrained response (:mod:`dqcopilot.ai.client`);
3. check the answer against the real data (:mod:`dqcopilot.ai.grounding`);
4. turn what survives into ordinary findings and proposals, tagged
   :data:`~dqcopilot.models.enums.FindingSource.AI`.

Step 4 is what keeps the guarantee: an AI suggestion becomes a normal proposal that the
user must approve, and the quality score ignores AI findings entirely, so the number on
screen stays reproducible whether or not an API key is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dqcopilot.ai.client import AiUnavailableError, AnthropicClient, build_user_content
from dqcopilot.ai.grounding import (
    ground_category_mappings,
    ground_column_interpretations,
    ground_suggested_rules,
)
from dqcopilot.ai.payload import category_payload, dataset_payload, findings_payload, should_mask
from dqcopilot.ai.schemas import (
    AnomalyExplanationResponse,
    CategoryMappingResponse,
    ColumnInterpretationResponse,
    SuggestedRule,
    SuggestedRuleResponse,
)
from dqcopilot.config import Settings, get_settings
from dqcopilot.logging_conf import get_logger
from dqcopilot.models.corrections import CorrectionProposal, make_proposal_id
from dqcopilot.models.enums import (
    CorrectionAction,
    FindingSource,
    IssueType,
    SemanticType,
    Severity,
)
from dqcopilot.models.findings import Finding, make_finding_id
from dqcopilot.services.analysis import AnalysisResult

logger = get_logger(__name__)

SHARED_SYSTEM = (
    "You are a data quality analyst reviewing a tabular dataset.\n"
    "You are given METADATA ONLY: column names, inferred types, counts, and a small "
    "number of example values. Some examples are masked, where letters appear as 'a' "
    "and digits as '9'; treat those as format information only.\n"
    "Rules you must follow:\n"
    "- Only refer to column names and values that appear in the input.\n"
    "- Never invent a value that is not shown.\n"
    "- If the evidence is weak, say so with a low confidence score rather than guessing.\n"
    "- Be specific and concise. Your output is reviewed by a person before anything "
    "is applied to the data."
)


@dataclass(slots=True)
class AiSuggestions:
    """Everything the AI produced for one analysis."""

    findings: list[Finding] = field(default_factory=list)
    proposals: list[CorrectionProposal] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    ran: bool = False

    @property
    def is_empty(self) -> bool:
        """True when nothing usable came back."""
        return not self.findings and not self.proposals


class AiSuggester:
    """Runs the AI tasks for one analysis."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: AnthropicClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or AnthropicClient(self.settings)

    @property
    def enabled(self) -> bool:
        """True when AI suggestions can run."""
        return self.client.enabled

    def run(self, analysis: AnalysisResult) -> AiSuggestions:
        """Run every AI task and collect the results.

        Failures in one task never stop the others, and never raise.
        """
        result = AiSuggestions()
        if not self.enabled:
            result.errors.append(
                "No ANTHROPIC_API_KEY is configured. Deterministic checks are unaffected."
            )
            return result

        result.ran = True
        for task in (
            self._interpret_columns,
            self._suggest_category_mappings,
            self._explain_anomalies,
            self._suggest_rules,
        ):
            try:
                task(analysis, result)
            except AiUnavailableError as exc:
                result.errors.append(str(exc))
                break
            except Exception:  # noqa: BLE001 - one broken task must not lose the rest
                logger.exception("AI task failed", extra={"task": task.__name__})
                result.errors.append("One AI task failed unexpectedly and was skipped.")

        result.usage = self.client.telemetry.summary()
        return result

    # ------------------------------------------------------------------ tasks

    def _interpret_columns(self, analysis: AnalysisResult, out: AiSuggestions) -> None:
        """Ask what ambiguous columns mean."""
        ambiguous = [
            column
            for column in analysis.profile.columns
            if column.semantic_type in (SemanticType.TEXT, SemanticType.UNKNOWN)
            or column.semantic_type_confidence < 0.8
        ]
        if not ambiguous:
            return

        payload = dataset_payload(analysis.profile, analysis.frame, self.settings)
        payload["columns"] = [
            entry
            for entry in payload["columns"]
            if entry["name"] in {column.name for column in ambiguous}
        ]
        if not payload["columns"]:
            return

        response = self.client.parse(
            task="column_meaning",
            system=(
                f"{SHARED_SYSTEM}\n\n"
                "Task: explain what each listed column most likely holds. Flag any column "
                "that probably contains personal data about individuals."
            ),
            user_content=build_user_content(payload),
            output_format=ColumnInterpretationResponse,
        )
        if not response.ok or response.data is None:
            if response.error:
                out.errors.append(response.error)
            return

        report = ground_column_interpretations(response.data.interpretations, analysis.profile)
        out.rejected.extend(report.rejection_reasons)

        for interpretation in report.accepted:
            out.findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.AI,
                        "ai_column_meaning",
                        IssueType.SCHEMA_MISMATCH,
                        interpretation.column,
                    ),
                    check_id="ai_column_meaning",
                    issue_type=IssueType.SCHEMA_MISMATCH,
                    source=FindingSource.AI,
                    severity=Severity.INFO,
                    column=interpretation.column,
                    title=f"AI reading of '{interpretation.column}'",
                    explanation=(
                        f"{interpretation.meaning}"
                        + (
                            f" Unit: {interpretation.likely_unit}."
                            if interpretation.likely_unit
                            else ""
                        )
                        + (
                            " This column probably holds personal data - treat it "
                            "accordingly before sharing the dataset."
                            if interpretation.is_personal_data
                            else ""
                        )
                        + f" (Model reasoning: {interpretation.reasoning})"
                    ),
                    affected_rows=0,
                    row_count=analysis.row_count,
                    confidence=interpretation.confidence,
                    details={
                        "personal_data": interpretation.is_personal_data,
                        "unit": interpretation.likely_unit,
                    },
                )
            )

    def _suggest_category_mappings(self, analysis: AnalysisResult, out: AiSuggestions) -> None:
        """Ask which category spellings mean the same thing."""
        candidates = [
            column
            for column in analysis.profile.columns
            if column.semantic_type is SemanticType.CATEGORICAL
            and 1 < column.unique_count <= 60
            and not should_mask(column)
        ]
        if not candidates:
            return

        for column in candidates[:5]:
            payload = category_payload(column, analysis.frame[column.name], self.settings)
            response = self.client.parse(
                task="category_mapping",
                system=(
                    f"{SHARED_SYSTEM}\n\n"
                    "Task: find values in this column that denote the same real-world "
                    "category written differently (for example an abbreviation and a full "
                    "name). For each, map the non-standard spelling onto the standard one. "
                    "Both values must already appear in the list. Map towards the more "
                    "frequent or more canonical form. Return nothing if the values are all "
                    "genuinely distinct categories."
                ),
                user_content=build_user_content(payload),
                output_format=CategoryMappingResponse,
            )
            if not response.ok or response.data is None:
                if response.error:
                    out.errors.append(response.error)
                continue

            report = ground_category_mappings(response.data.mappings, analysis.frame)
            out.rejected.extend(report.rejection_reasons)
            if not report.accepted:
                continue

            mapping = {item.from_value.strip(): item.to_value.strip() for item in report.accepted}
            affected = int(
                analysis.frame[column.name]
                .astype("string")
                .str.strip()
                .isin(mapping.keys())
                .fillna(False)
                .sum()
            )
            examples = "; ".join(
                f"'{item.from_value}' -> '{item.to_value}'" for item in report.accepted[:4]
            )

            finding = Finding(
                finding_id=make_finding_id(
                    FindingSource.AI,
                    "ai_category_mapping",
                    IssueType.INCONSISTENT_CATEGORY,
                    column.name,
                ),
                check_id="ai_category_mapping",
                issue_type=IssueType.INCONSISTENT_CATEGORY,
                source=FindingSource.AI,
                severity=Severity.INFO,
                column=column.name,
                title=f"AI suggests merging categories in '{column.name}'",
                explanation=(
                    f"The model proposes {len(report.accepted)} mapping(s) in "
                    f"'{column.name}': {examples}. Every value was checked against the "
                    "dataset before this was shown. This is a suggestion, not a finding: "
                    "it does not affect the quality score and nothing changes unless you "
                    "approve it."
                ),
                affected_rows=affected,
                row_count=analysis.row_count,
                confidence=min(item.confidence for item in report.accepted),
                details={"mapping_count": len(mapping)},
            )
            out.findings.append(finding)

            out.proposals.append(
                CorrectionProposal(
                    proposal_id=make_proposal_id(
                        finding.finding_id, CorrectionAction.MAP_CATEGORY, "ai"
                    ),
                    finding_id=finding.finding_id,
                    action=CorrectionAction.MAP_CATEGORY,
                    source=FindingSource.AI,
                    column=column.name,
                    title=f"Apply the AI category mapping to '{column.name}'",
                    description=(
                        f"Rewrite {affected:,} value(s) in '{column.name}' using the "
                        f"mapping the model proposed: {examples}. Read every mapping "
                        "before approving - the model is inferring meaning, not "
                        "checking a rule."
                    ),
                    rationale=finding.explanation,
                    parameters={"mapping": mapping},
                    affected_rows=affected,
                    confidence=finding.confidence,
                )
            )

    def _explain_anomalies(self, analysis: AnalysisResult, out: AiSuggestions) -> None:
        """Ask for plain-language explanations of the most severe findings."""
        severe = [
            finding
            for finding in analysis.findings.sorted()
            if finding.severity in (Severity.CRITICAL, Severity.HIGH)
        ][:8]
        if not severe:
            return

        response = self.client.parse(
            task="anomaly_explanation",
            system=(
                f"{SHARED_SYSTEM}\n\n"
                "Task: for each finding, explain in plain language what it means for "
                "someone who has to use this data, what most likely caused it, and what "
                "to do about it. Write for a business reader, not an engineer. Echo back "
                "the finding_reference you were given, unchanged."
            ),
            user_content=build_user_content(findings_payload(severe)),
            output_format=AnomalyExplanationResponse,
        )
        if not response.ok or response.data is None:
            if response.error:
                out.errors.append(response.error)
            return

        known = {finding.finding_id: finding for finding in severe}
        for explanation in response.data.explanations:
            source_finding = known.get(explanation.finding_reference)
            if source_finding is None:
                out.rejected.append(
                    f"Explanation referenced an unknown finding '{explanation.finding_reference}'."
                )
                continue

            out.findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.AI,
                        "ai_anomaly_explanation",
                        source_finding.issue_type,
                        source_finding.column,
                        fingerprint=source_finding.finding_id,
                    ),
                    check_id="ai_anomaly_explanation",
                    issue_type=source_finding.issue_type,
                    source=FindingSource.AI,
                    severity=Severity.INFO,
                    column=source_finding.column,
                    title=f"AI explanation: {source_finding.title}",
                    explanation=(
                        f"{explanation.plain_language}\n\n"
                        f"**Likely cause:** {explanation.likely_cause}\n\n"
                        f"**Suggested action:** {explanation.suggested_action}"
                        + (
                            f"\n\n**Business impact:** {explanation.business_impact}"
                            if explanation.business_impact
                            else ""
                        )
                    ),
                    affected_rows=source_finding.affected_rows,
                    row_count=source_finding.row_count,
                    confidence=0.7,
                    details={"explains_finding": source_finding.finding_id},
                )
            )

    def _suggest_rules(self, analysis: AnalysisResult, out: AiSuggestions) -> None:
        """Ask which business rules this dataset should have."""
        payload = dataset_payload(analysis.profile, analysis.frame, self.settings)
        existing = [rule.name for rule in analysis.rules.rules] if analysis.rules else []
        payload["existing_rule_names"] = existing

        response = self.client.parse(
            task="rule_suggestion",
            system=(
                f"{SHARED_SYSTEM}\n\n"
                "Task: propose data quality rules that a domain expert would want on this "
                "dataset and that are not already covered by existing_rule_names. Use only "
                "these rule types: not_null, range, allowed_values, no_future_dates, regex, "
                "unique. Propose at most eight rules, and only ones you can justify from "
                "the metadata shown."
            ),
            user_content=build_user_content(payload),
            output_format=SuggestedRuleResponse,
        )
        if not response.ok or response.data is None:
            if response.error:
                out.errors.append(response.error)
            return

        report = ground_suggested_rules(response.data.rules, analysis.frame)
        out.rejected.extend(report.rejection_reasons)

        for rule in report.accepted:
            if rule.name in existing:
                continue
            out.findings.append(
                Finding(
                    finding_id=make_finding_id(
                        FindingSource.AI,
                        "ai_rule_suggestion",
                        IssueType.BUSINESS_RULE_VIOLATION,
                        rule.column,
                        fingerprint=rule.name,
                    ),
                    check_id="ai_rule_suggestion",
                    issue_type=IssueType.BUSINESS_RULE_VIOLATION,
                    source=FindingSource.AI,
                    severity=Severity.INFO,
                    column=rule.column,
                    title=f"AI suggests a rule: {rule.name}",
                    explanation=(
                        f"{rule.description} Proposed as a `{rule.rule_type}` rule on "
                        f"'{rule.column}'. To adopt it, add it to "
                        "`config/business_rules.yaml` - suggested rules are never applied "
                        "automatically, so that the rule file stays something a human owns."
                    ),
                    affected_rows=0,
                    row_count=analysis.row_count,
                    confidence=rule.confidence,
                    details={
                        "rule_type": rule.rule_type,
                        "yaml": _rule_to_yaml(rule),
                    },
                )
            )


def _rule_to_yaml(rule: SuggestedRule) -> str:
    """Render a suggested rule as a YAML snippet the user can paste into the rule file."""
    lines = [
        f"- name: {rule.name}",
        f"  type: {rule.rule_type}",
        f"  column: {rule.column}",
        "  severity: medium",
    ]
    if rule.minimum is not None:
        lines.append(f"  minimum: {rule.minimum:g}")
    if rule.maximum is not None:
        lines.append(f"  maximum: {rule.maximum:g}")
    if rule.allowed_values:
        rendered = ", ".join(str(value) for value in rule.allowed_values)
        lines.append(f"  values: [{rendered}]")
    return "\n".join(lines)
