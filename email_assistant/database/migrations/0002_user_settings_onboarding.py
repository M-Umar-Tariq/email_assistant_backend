from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("database", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="usersettings",
            name="occupation",
            field=models.CharField(blank=True, default="", max_length=500),
        ),
        migrations.AddField(
            model_name="usersettings",
            name="important_emails_notes",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddField(
            model_name="usersettings",
            name="draft_style_notes",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddField(
            model_name="usersettings",
            name="ai_label_rules",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="usersettings",
            name="onboarding_completed",
            field=models.BooleanField(default=False),
        ),
    ]
