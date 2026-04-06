"""
Repair user_settings rows where `id` is null or missing.
Assigns a unique monotonic integer id to each affected row.

Usage:
    python manage.py fix_settings_ids          # dry-run (report only)
    python manage.py fix_settings_ids --apply  # actually fix
"""

from django.core.management.base import BaseCommand

from database.db import user_settings_col, next_user_settings_int_id


class Command(BaseCommand):
    help = "Assign integer id to user_settings rows where id is null/missing"

    def add_arguments(self, parser):
        parser.add_argument(
            "--apply",
            action="store_true",
            default=False,
            help="Actually update the database (default is dry-run)",
        )

    def handle(self, *args, **options):
        col = user_settings_col()
        broken = list(col.find({"$or": [{"id": None}, {"id": {"$exists": False}}]}))

        if not broken:
            self.stdout.write(self.style.SUCCESS("No user_settings rows with null/missing id found."))
            return

        self.stdout.write(f"Found {len(broken)} row(s) with null/missing id.")

        if not options["apply"]:
            for doc in broken:
                self.stdout.write(f"  [DRY-RUN] user_id={doc.get('user_id')}  _id={doc.get('_id')}")
            self.stdout.write(self.style.WARNING("Run with --apply to fix them."))
            return

        fixed = 0
        for doc in broken:
            new_id = next_user_settings_int_id()
            col.update_one({"_id": doc["_id"]}, {"$set": {"id": new_id}})
            self.stdout.write(f"  Fixed user_id={doc.get('user_id')} → id={new_id}")
            fixed += 1

        self.stdout.write(self.style.SUCCESS(f"Done. Fixed {fixed} row(s)."))
