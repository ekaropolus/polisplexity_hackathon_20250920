from __future__ import annotations
from django.contrib import admin
from django.utils.html import format_html
from .models import SiteRun

@admin.register(SiteRun)
class SiteRunAdmin(admin.ModelAdmin):
    list_display = ("created_at", "city", "business", "short_id", "preview", "download")
    list_filter = ("city", "business", "created_at")
    search_fields = ("search_id", "city", "business")
    readonly_fields = ("created_at", "search_id", "city", "business", "payload_json", "result_json")

    def short_id(self, obj: SiteRun) -> str:
        return obj.search_id[:8]

    def preview(self, obj: SiteRun):
        if obj.map_url:
            return format_html('<a href="{}" target="_blank">map</a>', obj.map_url)
        return "—"

    def download(self, obj: SiteRun):
        # if you added a PNG/CSV download endpoint, link it here later
        url = (obj.result_json or {}).get("download_url")
        if url:
            return format_html('<a href="{}" target="_blank">download</a>', url)
        return "—"
