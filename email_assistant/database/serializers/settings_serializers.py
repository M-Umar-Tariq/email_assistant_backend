from rest_framework import serializers


class SettingsUpdateSerializer(serializers.Serializer):
    daily_briefing = serializers.BooleanField(required=False)
    slack_digest = serializers.BooleanField(required=False)
    critical_alerts = serializers.BooleanField(required=False)
    ai_suggestions = serializers.BooleanField(required=False)
    auto_labeling = serializers.BooleanField(required=False)
    thread_summaries = serializers.BooleanField(required=False)
    sync_range_months = serializers.IntegerField(required=False, min_value=1, max_value=60)


class SettingsResponseSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    daily_briefing = serializers.BooleanField()
    slack_digest = serializers.BooleanField()
    critical_alerts = serializers.BooleanField()
    ai_suggestions = serializers.BooleanField()
    auto_labeling = serializers.BooleanField()
    thread_summaries = serializers.BooleanField()
    sync_range_months = serializers.IntegerField()
