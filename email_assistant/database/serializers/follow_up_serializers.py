from rest_framework import serializers


class FollowUpCreateSerializer(serializers.Serializer):
    email_id = serializers.CharField()
    due_date = serializers.DateTimeField()
    suggested_action = serializers.CharField(required=False, default="")


class FollowUpUpdateSerializer(serializers.Serializer):
    status = serializers.ChoiceField(
        choices=["pending", "overdue", "completed", "snoozed"],
        required=False,
    )
    due_date = serializers.DateTimeField(required=False)
    suggested_action = serializers.CharField(required=False)


class FollowUpResponseSerializer(serializers.Serializer):
    id = serializers.CharField()
    user_id = serializers.CharField()
    email_id = serializers.CharField()
    due_date = serializers.DateTimeField()
    status = serializers.CharField()
    auto_reminder_sent = serializers.BooleanField()
    suggested_action = serializers.CharField()
    days_waiting = serializers.IntegerField()
    created_at = serializers.DateTimeField()
