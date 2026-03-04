"""Refresh tokens collection (TTL index in db.ensure_indexes)."""

from django.db import models


class RefreshToken(models.Model):
    """Django model for MongoDB `refresh_tokens` collection (migrations/ORM)."""
    user_id = models.CharField(max_length=255, db_index=True)
    token = models.CharField(max_length=512)
    expires_at = models.DateTimeField()

    class Meta:
        db_table = "refresh_tokens"
        managed = True
