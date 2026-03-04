from django.db import models


class EmailMetadata(models.Model):
    """Django model for MongoDB `email_metadata` collection (migrations/ORM)."""
    user_id = models.CharField(max_length=255, db_index=True)
    mailbox_id = models.CharField(max_length=255, db_index=True)
    message_id = models.CharField(max_length=512)
    thread_id = models.CharField(max_length=512, null=True, blank=True)
    subject = models.CharField(max_length=1024, default="")
    from_name = models.CharField(max_length=255, default="")
    from_email = models.CharField(max_length=255, default="")
    to = models.JSONField(default=list)
    date = models.DateTimeField(auto_now_add=True)
    preview = models.TextField(default="")
    read = models.BooleanField(default=False)
    starred = models.BooleanField(default=False)
    labels = models.JSONField(default=list)
    has_attachment = models.BooleanField(default=False)
    priority = models.CharField(max_length=32, default="medium")
    category = models.CharField(max_length=128, null=True, blank=True)
    ai_summary = models.TextField(null=True, blank=True)
    sentiment_score = models.FloatField(null=True, blank=True)
    snoozed_until = models.DateTimeField(null=True, blank=True)
    archived = models.BooleanField(default=False)
    trashed = models.BooleanField(default=False)
    total_chunks = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "email_metadata"
        managed = True
