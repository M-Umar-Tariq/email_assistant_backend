from django.db import models


class FollowUp(models.Model):
    """Django model for MongoDB `follow_ups` collection (migrations/ORM)."""
    user_id = models.CharField(max_length=255, db_index=True)
    email_id = models.CharField(max_length=255)
    due_date = models.DateTimeField()
    status = models.CharField(max_length=64, default="pending")
    auto_reminder_sent = models.BooleanField(default=False)
    days_waiting = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "follow_ups"
        managed = True
