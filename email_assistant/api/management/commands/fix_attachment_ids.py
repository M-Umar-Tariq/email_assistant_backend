"""
Repair email_attachments rows where `id` is null or missing (Djongo unique index).

Usage:
    python manage.py fix_attachment_ids          # dry-run
    python manage.py fix_attachment_ids --apply  # assign ids
"""

from django.core.management.base import BaseCommand

from database.db import email_attachments_col, next_email_attachment_int_id


class Command(BaseCommand):
    help = "Assign integer id to email_attachments rows where id is null/missing"

    def add_arguments(self, parser):
        parser.add_argument(
            "--apply",
            action="store_true",
            default=False,
            help="Actually update the database (default is dry-run)",
        )

    def handle(self, *args, **options):
        col = email_attachments_col()
        broken = list(col.find({"$or": [{"id": None}, {"id": {"$exists": False}}]}))

        if not broken:
            self.stdout.write(self.style.SUCCESS("No email_attachments rows with null/missing id found."))
            return

        self.stdout.write(f"Found {len(broken)} row(s) with null/missing id.")

        if not options["apply"]:
            for doc in broken:
                self.stdout.write(
                    f"  [DRY-RUN] email_id={doc.get('email_id')} index={doc.get('index')} _id={doc.get('_id')}"
                )
            self.stdout.write(self.style.WARNING("Run with --apply to fix them."))
            return

        fixed = 0
        for doc in broken:
            new_id = next_email_attachment_int_id()
            col.update_one({"_id": doc["_id"]}, {"$set": {"id": new_id}})
            self.stdout.write(
                f"  Fixed email_id={doc.get('email_id')} index={doc.get('index')} -> id={new_id}"
            )
            fixed += 1

        self.stdout.write(self.style.SUCCESS(f"Done. Fixed {fixed} row(s)."))
