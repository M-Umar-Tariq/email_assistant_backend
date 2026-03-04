from django.db import models


class Mailbox(models.Model):
    """Django model for MongoDB `mailboxes` collection (migrations/ORM)."""
    user_id = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    email = models.CharField(max_length=255)
    color = models.CharField(max_length=32, default="#0ea5e9")
    imap_host = models.CharField(max_length=255)
    imap_port = models.IntegerField(default=993)
    imap_secure = models.BooleanField(default=True)
    smtp_host = models.CharField(max_length=255)
    smtp_port = models.IntegerField(default=587)
    smtp_secure = models.BooleanField(default=True)
    username = models.CharField(max_length=255)
    encrypted_password = models.CharField(max_length=1024)
    last_sync_at = models.DateTimeField(null=True, blank=True)
    sync_status = models.CharField(max_length=64, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "mailboxes"
        managed = True
