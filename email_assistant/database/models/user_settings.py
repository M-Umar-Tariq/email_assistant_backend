from django.db import models


class UserSettings(models.Model):
    """Django model for MongoDB `user_settings` collection (migrations/ORM)."""

    user_id = models.CharField(max_length=255, db_index=True)
    daily_briefing = models.BooleanField(default=True)
    slack_digest = models.BooleanField(default=False)
    critical_alerts = models.BooleanField(default=True)
    ai_suggestions = models.BooleanField(default=True)
    auto_labeling = models.BooleanField(default=True)
    thread_summaries = models.BooleanField(default=True)
    sync_range_months = models.IntegerField(default=12)
    updated_at = models.DateTimeField(auto_now=True)
    # Onboarding / personalization (optional; used by Smart Mail AI wizard)
    occupation = models.CharField(max_length=500, blank=True, default="")
    important_emails_notes = models.TextField(blank=True, default="")
    draft_style_notes = models.TextField(blank=True, default="")
    ai_label_rules = models.JSONField(default=list, blank=True)
    onboarding_completed = models.BooleanField(default=False)

    class Meta:
        db_table = "user_settings"
        managed = True
