"""Alert management for HeliosLM production."""
import time
from typing import List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class AlertSeverity(Enum):
    WARNING = "warning"
    CRITICAL = "critical"
    INFO = "info"


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    expr: str
    severity: AlertSeverity
    duration: int  # seconds before firing
    summary: str
    description: str


@dataclass
class Alert:
    """Fired alert instance."""
    rule: AlertRule
    value: float
    fired_at: float
    resolved: bool = False


class AlertManager:
    """
    Manage alert rules and notifications.

    Supports multiple notification channels:
      - Slack
      - PagerDuty
      - Email
      - Webhook
    """

    def __init__(self):
        self.rules: List[AlertRule] = []
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.notifiers: List[Callable] = []

    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules.append(rule)

    def add_notifier(self, notifier: Callable):
        """Add a notification handler."""
        self.notifiers.append(notifier)

    def evaluate(self, metrics: dict):
        """Evaluate all rules against current metrics."""
        now = time.time()

        for rule in self.rules:
            # Evaluate expression (simplified)
            value = self._eval_expr(rule.expr, metrics)

            if value:
                # Check if already firing
                existing = [a for a in self.active_alerts if a.rule.name == rule.name and not a.resolved]
                if not existing:
                    alert = Alert(rule=rule, value=value, fired_at=now)
                    self.active_alerts.append(alert)
                    self._notify(alert)
            else:
                # Resolve if firing
                for alert in self.active_alerts:
                    if alert.rule.name == rule.name and not alert.resolved:
                        alert.resolved = True
                        self.alert_history.append(alert)

    def _eval_expr(self, expr: str, metrics: dict) -> float:
        """Evaluate alert expression against metrics."""
        # Simplified: expressions like "gpu_util > 0.9"
        try:
            parts = expr.split()
            metric_name = parts[0]
            operator = parts[1]
            threshold = float(parts[2])

            value = metrics.get(metric_name, 0)

            if operator == ">":
                return value if value > threshold else 0
            elif operator == "<":
                return value if value < threshold else 0
            elif operator == ">=":
                return value if value >= threshold else 0
            elif operator == "<=":
                return value if value <= threshold else 0
            return 0
        except:
            return 0

    def _notify(self, alert: Alert):
        """Send notifications for an alert."""
        message = f"[{alert.rule.severity.value.upper()}] {alert.rule.name}\n{alert.rule.description}\nValue: {alert.value:.2f}"

        for notifier in self.notifiers:
            try:
                notifier(message)
            except Exception as e:
                print(f"Notification failed: {e}")

    def get_default_rules(self) -> List[AlertRule]:
        """Get default alert rules for HeliosLM."""
        return [
            AlertRule(
                name="HighGPUUtilization",
                expr="gpu_util > 0.95",
                severity=AlertSeverity.WARNING,
                duration=300,
                summary="GPU utilization is very high",
                description="GPU utilization has been above 95% for 5 minutes. Consider scaling up.",
            ),
            AlertRule(
                name="HighErrorRate",
                expr="error_rate > 0.05",
                severity=AlertSeverity.CRITICAL,
                duration=60,
                summary="Error rate is elevated",
                description="Error rate has exceeded 5%. Immediate investigation required.",
            ),
            AlertRule(
                name="HighQueueDepth",
                expr="queue_depth > 100",
                severity=AlertSeverity.WARNING,
                duration=120,
                summary="Request queue is backing up",
                description="Queue depth has exceeded 100 requests. Scaling may be needed.",
            ),
            AlertRule(
                name="HighLatency",
                expr="p99_latency_ms > 10000",
                severity=AlertSeverity.WARNING,
                duration=300,
                summary="P99 latency is high",
                description="P99 latency has exceeded 10 seconds. Check model performance.",
            ),
        ]
