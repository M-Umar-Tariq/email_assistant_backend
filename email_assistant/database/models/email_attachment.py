"""Email attachments collection."""

from django.db import models


class EmailAttachment(models.Model):
    """Django model for MongoDB `email_attachments` collection (migrations/ORM)."""
    user_id = models.CharField(max_length=255, db_index=True)
    email_id = models.CharField(max_length=255)
    filename = models.CharField(max_length=512)
    content_type = models.CharField(max_length=128, default="application/octet-stream")
    size = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "email_attachments"
        managed = True
