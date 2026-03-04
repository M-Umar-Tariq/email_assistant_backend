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

    class Meta:
        db_table = "user_settings"
        managed = True
