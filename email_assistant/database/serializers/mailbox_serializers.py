from rest_framework import serializers


class MailboxCreateSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=100)
    email = serializers.EmailField()
    color = serializers.CharField(max_length=20, default="#0ea5e9")
    imap_host = serializers.CharField()
    imap_port = serializers.IntegerField(default=993)
    imap_secure = serializers.BooleanField(default=True)
    smtp_host = serializers.CharField()
    smtp_port = serializers.IntegerField(default=587)
    smtp_secure = serializers.BooleanField(default=True)
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)


class MailboxUpdateSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=100, required=False)
    color = serializers.CharField(max_length=20, required=False)


class MailboxResponseSerializer(serializers.Serializer):
    id = serializers.CharField()
    name = serializers.CharField()
    email = serializers.EmailField()
    color = serializers.CharField()
    imap_host = serializers.CharField()
    imap_port = serializers.IntegerField()
    smtp_host = serializers.CharField()
    smtp_port = serializers.IntegerField()
    username = serializers.CharField()
    last_sync_at = serializers.DateTimeField(allow_null=True)
    sync_status = serializers.CharField()
    created_at = serializers.DateTimeField()
