from django.urls import path, include

from database.db import ensure_qdrant_collection, ensure_indexes, reset_stale_syncs
try:
    ensure_qdrant_collection()
except Exception as e:
    print(f"[STARTUP] Qdrant collection setup failed: {e}")
try:
    ensure_indexes()
except Exception as e:
    print(f"[STARTUP] MongoDB index setup failed: {e}")
try:
    reset_stale_syncs()
except Exception as e:
    print(f"[STARTUP] Sync status reset failed: {e}")

urlpatterns = [
    path("admin/", include("api.controllers.admin.urls")),
    path("auth/", include("api.controllers.auth.urls")),
    path("mailboxes/", include("api.controllers.mailboxes.urls")),
    path("emails/", include("api.controllers.emails.urls")),
    path("follow-ups/", include("api.controllers.follow_ups.urls")),
    path("briefing/", include("api.controllers.briefing.urls")),
    path("analytics/", include("api.controllers.analytics.urls")),
    path("ai/", include("api.controllers.ai.urls")),
    path("agent/", include("api.controllers.agent.urls")),
    path("compose/", include("api.controllers.compose.urls")),
    path("settings/", include("api.controllers.settings.urls")),
    path("search/", include("api.controllers.search.urls")),
    path("calendar/", include("api.controllers.calendar.urls")),
    path("feedback/", include("api.controllers.feedback.urls")),
]
