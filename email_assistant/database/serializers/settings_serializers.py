from rest_framework import serializers


class SettingsUpdateSerializer(serializers.Serializer):
    daily_briefing = serializers.BooleanField(required=False)
    slack_digest = serializers.BooleanField(required=False)
    critical_alerts = serializers.BooleanField(required=False)
    ai_suggestions = serializers.BooleanField(required=False)
    auto_labeling = serializers.BooleanField(required=False)
    thread_summaries = serializers.BooleanField(required=False)
    sync_range_months = serializers.IntegerField(required=False, min_value=1, max_value=60)
    occupation = serializers.CharField(required=False, allow_blank=True, max_length=500)
    important_emails_notes = serializers.CharField(required=False, allow_blank=True, max_length=4000)
    draft_style_notes = serializers.CharField(required=False, allow_blank=True, max_length=4000)
    ai_label_rules = serializers.ListField(
        child=serializers.DictField(),
        required=False,
        max_length=12,
    )
    onboarding_completed = serializers.BooleanField(required=False)

    def validate_ai_label_rules(self, value):
        if not value:
            return []
        out = []
        for item in value[:12]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()[:128]
            instruction = str(item.get("instruction", "")).strip()[:2000]
            if name or instruction:
                out.append({"name": name, "instruction": instruction})
        return out


class SettingsResponseSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    daily_briefing = serializers.BooleanField()
    slack_digest = serializers.BooleanField()
    critical_alerts = serializers.BooleanField()
    ai_suggestions = serializers.BooleanField()
    auto_labeling = serializers.BooleanField()
    thread_summaries = serializers.BooleanField()
    sync_range_months = serializers.IntegerField()
