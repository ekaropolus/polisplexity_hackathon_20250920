from __future__ import annotations
from django.db import models

class SiteRun(models.Model):
    search_id = models.CharField(max_length=64, db_index=True)
    city = models.CharField(max_length=64)
    business = models.CharField(max_length=128)
    payload_json = models.JSONField()   # request we received
    result_json = models.JSONField()    # full response we returned
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.created_at:%Y-%m-%d %H:%M} — {self.city}/{self.business} — {self.search_id[:8]}"

    # convenience accessors
    @property
    def map_url(self) -> str | None:
        return (self.result_json or {}).get("map_url")

    @property
    def geojson_url(self) -> str | None:
        # if you already expose one, wire it here later
        return (self.result_json or {}).get("geojson_url")
