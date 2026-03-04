from rest_framework import serializers


class EmailListResponseSerializer(serializers.Serializer):
    id = serializers.CharField()
    mailbox_id = serializers.CharField()
    subject = serializers.CharField()
    from_name = serializers.CharField()
    from_email = serializers.CharField()
    to = serializers.ListField()
    date = serializers.DateTimeField()
    preview = serializers.CharField()
    read = serializers.BooleanField()
    starred = serializers.BooleanField()
    labels = serializers.ListField(child=serializers.CharField())
    has_attachment = serializers.BooleanField()
    priority = serializers.CharField()
    category = serializers.CharField(allow_null=True)
    ai_summary = serializers.CharField(allow_null=True)
    sentiment_score = serializers.FloatField(allow_null=True)
    snoozed_until = serializers.DateTimeField(allow_null=True)


class EmailDetailResponseSerializer(EmailListResponseSerializer):
    body = serializers.CharField()
    thread_id = serializers.CharField(allow_null=True)
    total_chunks = serializers.IntegerField()


class EmailUpdateSerializer(serializers.Serializer):
    read = serializers.BooleanField(required=False)
    starred = serializers.BooleanField(required=False)
    labels = serializers.ListField(child=serializers.CharField(), required=False)


class EmailSnoozeSerializer(serializers.Serializer):
    hours = serializers.IntegerField(min_value=1)


class EmailSendSerializer(serializers.Serializer):
    mailbox_id = serializers.CharField()
    to = serializers.ListField(child=serializers.EmailField())
    cc = serializers.ListField(child=serializers.EmailField(), required=False, default=[])
    subject = serializers.CharField()
    body = serializers.CharField()
    reply_to_email_id = serializers.CharField(required=False, allow_blank=True)


class EmailForwardSerializer(serializers.Serializer):
    mailbox_id = serializers.CharField()
    to = serializers.ListField(child=serializers.EmailField())
    body = serializers.CharField(required=False, default="")
