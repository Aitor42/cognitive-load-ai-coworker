"""iCalendar (.ics) parsing and absence detection for LoadGuard.

Extracts meeting events and out-of-office / leave absences from RFC 5545
calendars without external dependencies.
"""

from __future__ import annotations

import logging
import re
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .models import LEAVE, VACATION, Absence, Event

logger = logging.getLogger(__name__)

ISO = "%Y-%m-%dT%H:%M:%SZ"

_DURATION_RE = re.compile(
    r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?"
)
_VEVENT_BLOCK_RE = re.compile(r"BEGIN:VEVENT(.*?)END:VEVENT", re.DOTALL)

# Summary substrings/keywords (lowercased) that mark a calendar event as an absence.
ABSENCE_SUMMARY_KEYWORDS = (
    "out of office",
    "out-of-office",
    "ooo",
    "vacation",
    "vacaciones",
    "on leave",
    "annual leave",
    "baja",
    "permiso",
)
VACATION_KEYWORDS = ("vacation", "vacaciones")

MAX_RRULE_OCCURRENCES = 366

_WEEKDAY_CODES = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}

_WINDOWS_TZ_MAP: dict[str, str] = {
    "Dateline Standard Time": "Etc/GMT+12",
    "UTC-11": "Etc/GMT+11",
    "Aleutian Standard Time": "America/Adak",
    "Hawaiian Standard Time": "Pacific/Honolulu",
    "Marquesas Standard Time": "Pacific/Marquesas",
    "Alaskan Standard Time": "America/Anchorage",
    "UTC-09": "Etc/GMT+9",
    "Pacific Standard Time (Mexico)": "America/Tijuana",
    "UTC-08": "Etc/GMT+8",
    "Pacific Standard Time": "America/Los_Angeles",
    "US Mountain Standard Time": "America/Phoenix",
    "Mountain Standard Time (Mexico)": "America/Mazatlan",
    "Mountain Standard Time": "America/Denver",
    "Yukon Standard Time": "America/Whitehorse",
    "Central America Standard Time": "America/Guatemala",
    "Central Standard Time": "America/Chicago",
    "Easter Island Standard Time": "Pacific/Easter",
    "Central Standard Time (Mexico)": "America/Mexico_City",
    "Canada Central Standard Time": "America/Regina",
    "SA Pacific Standard Time": "America/Bogota",
    "Eastern Standard Time (Mexico)": "America/Cancun",
    "Eastern Standard Time": "America/New_York",
    "Haiti Standard Time": "America/Port-au-Prince",
    "Cuba Standard Time": "America/Havana",
    "US Eastern Standard Time": "America/Indiana/Indianapolis",
    "Turks And Caicos Standard Time": "America/Grand_Turk",
    "Paraguay Standard Time": "America/Asuncion",
    "Atlantic Standard Time": "America/Halifax",
    "Venezuela Standard Time": "America/Caracas",
    "Central Brazilian Standard Time": "America/Cuiaba",
    "SA Western Standard Time": "America/La_Paz",
    "Pacific SA Standard Time": "America/Santiago",
    "Newfoundland Standard Time": "America/St_Johns",
    "Tocantins Standard Time": "America/Araguaina",
    "E. South America Standard Time": "America/Sao_Paulo",
    "SA Eastern Standard Time": "America/Cayenne",
    "Argentina Standard Time": "America/Buenos_Aires",
    "Greenland Standard Time": "America/Nuuk",
    "Montevideo Standard Time": "America/Montevideo",
    "Magallanes Standard Time": "America/Punta_Arenas",
    "Saint Pierre Standard Time": "America/Miquelon",
    "Bahia Standard Time": "America/Bahia",
    "UTC-02": "Etc/GMT+2",
    "Azores Standard Time": "Atlantic/Azores",
    "Cape Verde Standard Time": "Atlantic/Cape_Verde",
    "UTC": "Etc/UTC",
    "GMT Standard Time": "Europe/London",
    "Greenwich Standard Time": "Atlantic/Reykjavik",
    "Sao Tome Standard Time": "Africa/Sao_Tome",
    "Morocco Standard Time": "Africa/Casablanca",
    "W. Europe Standard Time": "Europe/Berlin",
    "Central Europe Standard Time": "Europe/Budapest",
    "Romance Standard Time": "Europe/Paris",
    "Central European Standard Time": "Europe/Warsaw",
    "W. Central Africa Standard Time": "Africa/Lagos",
    "Jordan Standard Time": "Asia/Amman",
    "GTB Standard Time": "Europe/Bucharest",
    "Middle East Standard Time": "Asia/Beirut",
    "Egypt Standard Time": "Africa/Cairo",
    "E. Europe Standard Time": "Europe/Chisinau",
    "Syria Standard Time": "Asia/Damascus",
    "West Bank Standard Time": "Asia/Hebron",
    "South Africa Standard Time": "Africa/Johannesburg",
    "FLE Standard Time": "Europe/Kyiv",
    "Israel Standard Time": "Asia/Jerusalem",
    "South Sudan Standard Time": "Africa/Juba",
    "Kaliningrad Standard Time": "Europe/Kaliningrad",
    "Sudan Standard Time": "Africa/Khartoum",
    "Libya Standard Time": "Africa/Tripoli",
    "Namibia Standard Time": "Africa/Windhoek",
    "Arabic Standard Time": "Asia/Baghdad",
    "Turkey Standard Time": "Europe/Istanbul",
    "Arab Standard Time": "Asia/Riyadh",
    "Belarus Standard Time": "Europe/Minsk",
    "Russian Standard Time": "Europe/Moscow",
    "E. Africa Standard Time": "Africa/Nairobi",
    "Iran Standard Time": "Asia/Tehran",
    "Arabian Standard Time": "Asia/Dubai",
    "Astrakhan Standard Time": "Europe/Astrakhan",
    "Azerbaijan Standard Time": "Asia/Baku",
    "Russia Time Zone 3": "Europe/Samara",
    "Mauritius Standard Time": "Indian/Mauritius",
    "Saratov Standard Time": "Europe/Saratov",
    "Georgian Standard Time": "Asia/Tbilisi",
    "Volgograd Standard Time": "Europe/Volgograd",
    "Caucasus Standard Time": "Asia/Yerevan",
    "Afghanistan Standard Time": "Asia/Kabul",
    "West Asia Standard Time": "Asia/Tashkent",
    "Ekaterinburg Standard Time": "Asia/Yekaterinburg",
    "Pakistan Standard Time": "Asia/Karachi",
    "Qyzylorda Standard Time": "Asia/Qyzylorda",
    "India Standard Time": "Asia/Kolkata",
    "Sri Lanka Standard Time": "Asia/Colombo",
    "Nepal Standard Time": "Asia/Kathmandu",
    "Central Asia Standard Time": "Asia/Bishkek",
    "Bangladesh Standard Time": "Asia/Dhaka",
    "Omsk Standard Time": "Asia/Omsk",
    "Myanmar Standard Time": "Asia/Yangon",
    "SE Asia Standard Time": "Asia/Bangkok",
    "Altai Standard Time": "Asia/Barnaul",
    "W. Mongolia Standard Time": "Asia/Hovd",
    "North Asia Standard Time": "Asia/Krasnoyarsk",
    "N. Central Asia Standard Time": "Asia/Novosibirsk",
    "Tomsk Standard Time": "Asia/Tomsk",
    "China Standard Time": "Asia/Shanghai",
    "North Asia East Standard Time": "Asia/Irkutsk",
    "Singapore Standard Time": "Asia/Singapore",
    "W. Australia Standard Time": "Australia/Perth",
    "Taipei Standard Time": "Asia/Taipei",
    "Ulaanbaatar Standard Time": "Asia/Ulaanbaatar",
    "Aus Central W. Standard Time": "Australia/Eucla",
    "Transbaikal Standard Time": "Asia/Chita",
    "Tokyo Standard Time": "Asia/Tokyo",
    "North Korea Standard Time": "Asia/Pyongyang",
    "Korea Standard Time": "Asia/Seoul",
    "Yakutsk Standard Time": "Asia/Yakutsk",
    "Cen. Australia Standard Time": "Australia/Adelaide",
    "AUS Central Standard Time": "Australia/Darwin",
    "E. Australia Standard Time": "Australia/Brisbane",
    "AUS Eastern Standard Time": "Australia/Sydney",
    "West Pacific Standard Time": "Pacific/Port_Moresby",
    "Tasmania Standard Time": "Australia/Hobart",
    "Vladivostok Standard Time": "Asia/Vladivostok",
    "Lord Howe Standard Time": "Australia/Lord_Howe",
    "Bougainville Standard Time": "Pacific/Bougainville",
    "Russia Time Zone 10": "Asia/Srednekolymsk",
    "Magadan Standard Time": "Asia/Magadan",
    "Norfolk Standard Time": "Pacific/Norfolk",
    "Sakhalin Standard Time": "Asia/Sakhalin",
    "Central Pacific Standard Time": "Pacific/Guadalcanal",
    "Russia Time Zone 11": "Asia/Kamchatka",
    "New Zealand Standard Time": "Pacific/Auckland",
    "UTC+12": "Etc/GMT-12",
    "Fiji Standard Time": "Pacific/Fiji",
    "Chatham Islands Standard Time": "Pacific/Chatham",
    "UTC+13": "Etc/GMT-13",
    "Tonga Standard Time": "Pacific/Tongatapu",
    "Samoa Standard Time": "Pacific/Apia",
    "Line Islands Standard Time": "Pacific/Kiritimati",
}


def _parse_ics_datetime(value: str, tzid: str | None = None) -> float:
    """Parse an ICS DATE-TIME value to epoch seconds (UTC)."""
    value = value.strip()
    if value.endswith("Z"):
        dt = datetime.strptime(value[:-1], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%S%z").timestamp()
    except ValueError:
        pass
    dt = datetime.strptime(value, "%Y%m%dT%H%M%S")
    if tzid:
        try:
            return dt.replace(tzinfo=ZoneInfo(_WINDOWS_TZ_MAP.get(tzid, tzid))).timestamp()
        except (ZoneInfoNotFoundError, ValueError, OSError):
            pass
    return dt.replace(tzinfo=timezone.utc).timestamp()


def _parse_ics_date(value: str) -> float:
    """Parse an all-day ICS DATE value (YYYYMMDD) to epoch seconds (UTC midnight)."""
    dt = datetime.strptime(value.strip(), "%Y%m%d").replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _parse_iso(value: str) -> float:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _parse_duration(value: str) -> float:
    m = _DURATION_RE.match(value.strip())
    if not m or not any(m.groups()):
        return 60.0
    days = int(m.group("days") or 0)
    h = int(m.group("hours") or 0)
    minutes = int(m.group("minutes") or 0)
    s = int(m.group("seconds") or 0)
    return days * 1440 + h * 60 + minutes + s / 60.0


def _unfold(text: str) -> str:
    lines = text.splitlines()
    unfolded: list[str] = []
    for line in lines:
        if line.startswith(" ") and unfolded:
            unfolded[-1] += line[1:]
        else:
            unfolded.append(line)
    return "\n".join(unfolded)


def _vevents_text(text: str) -> list[tuple[dict[str, str], dict[str, str | None], bool]]:
    out: list[tuple[dict[str, str], dict[str, str | None], bool]] = []
    for block in _VEVENT_BLOCK_RE.findall(_unfold(text)):
        props: dict[str, str] = {}
        tzids: dict[str, str | None] = {}
        all_day = False
        for line in block.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            name_part, _, value = line.partition(":")
            if "VALUE=DATE" in name_part.upper():
                all_day = True
            name = name_part.split(";")[0].upper()
            tzid = next(
                (
                    p.split("=", 1)[1]
                    for p in name_part.split(";")[1:]
                    if p.upper().startswith("TZID=")
                ),
                None,
            )
            if name in props:
                props[name] += "," + value.strip()
            else:
                props[name] = value.strip()
            tzids[name] = tzid
        if "DTSTART" not in props:
            continue
        out.append((props, tzids, all_day))
    return out


def _vevents(path: Path) -> list[tuple[dict[str, str], dict[str, str | None], bool]]:
    return _vevents_text(path.read_text(encoding="utf-8"))


def _is_absence(props: dict[str, str], all_day: bool = False) -> bool:
    """Detect an out-of-office / vacation / leave event from its properties."""
    summary = props.get("SUMMARY", "").lower().strip()
    for k in ABSENCE_SUMMARY_KEYWORDS:
        if k in summary:
            if (
                all_day
                or summary == k
                or summary.startswith(f"{k} ")
                or summary.startswith(f"{k}:")
                or summary.startswith(f"{k}-")
                or f"[{k}]" in summary
                or f"({k})" in summary
                or summary.endswith(f" {k}")
                or "on leave" in summary
                or "out of office" in summary
                or "annual leave" in summary
            ):
                if (
                    k == "permiso"
                    and not all_day
                    and summary != "permiso"
                    and not summary.startswith("permiso retribuido")
                    and not summary.startswith("permiso por")
                    and not summary.startswith("permiso médico")
                ):
                    continue
                return True
    if props.get("X-MICROSOFT-CDO-BUSYSTATUS", "").upper() == "OOF":
        return True
    return all_day and props.get("TRANSP", "").upper() == "TRANSPARENT"


def _absence_kind(props: dict[str, str]) -> str:
    summary = props.get("SUMMARY", "").lower()
    if any(k in summary for k in VACATION_KEYWORDS):
        return VACATION
    return LEAVE


def _absence_end(
    props: dict[str, str],
    start: float,
    all_day: bool,
    tzids: dict[str, str | None] | None = None,
) -> float:
    tzids = tzids or {}
    if "DTEND" in props:
        if all_day:
            return _parse_ics_date(props["DTEND"]) - 1.0
        return _parse_ics_datetime(props["DTEND"], tzids.get("DTEND"))
    if all_day:
        return start + 86400.0 - 1.0
    if "DURATION" in props:
        return start + _parse_duration(props["DURATION"]) * 60.0
    return start + 3600.0


def _parse_rrule(value: str) -> dict[str, str]:
    return {
        part.split("=", 1)[0]: part.split("=", 1)[1] for part in value.split(";") if "=" in part
    }


def _parse_ics_date_or_datetime(value: str, tzid: str | None = None) -> float:
    value = value.strip()
    if len(value) == 8 and value.isdigit():
        return _parse_ics_date(value)
    return _parse_ics_datetime(value, tzid)


def _expand_rrule(
    start_epoch: float,
    rrule: dict[str, str],
    until_epoch: float | None = None,
    exdates: set[float] | None = None,
) -> list[float]:
    freq = rrule.get("FREQ", "").upper()
    if freq not in ("DAILY", "WEEKLY", "MONTHLY", "YEARLY"):
        return [start_epoch]
    try:
        interval = max(int(rrule.get("INTERVAL", "1")), 1)
    except ValueError:
        interval = 1
    try:
        count = int(rrule["COUNT"]) if "COUNT" in rrule else None
    except ValueError:
        count = None
    byday = [d.upper() for d in rrule.get("BYDAY", "").split(",") if d]
    if byday and not all(d in _WEEKDAY_CODES for d in byday):
        byday = []
    limit = count if count is not None else MAX_RRULE_OCCURRENCES

    start_dt = datetime.fromtimestamp(start_epoch, tz=timezone.utc)
    start_date = start_dt.date()

    def ts(d: date) -> float:
        return datetime(
            d.year,
            d.month,
            d.day,
            start_dt.hour,
            start_dt.minute,
            start_dt.second,
            tzinfo=timezone.utc,
        ).timestamp()

    days: list[date] = []
    if freq == "DAILY":
        for i in range(limit):
            d = start_date + timedelta(days=i * interval)
            if until_epoch is not None and ts(d) > until_epoch:
                break
            days.append(d)
    elif freq == "WEEKLY":
        week_start = start_date - timedelta(days=start_date.weekday())
        if byday:
            offsets = sorted(_WEEKDAY_CODES[b] for b in byday)
            week = 0
            while len(days) < limit:
                base = week_start + timedelta(weeks=week * interval)
                for off in offsets:
                    d = base + timedelta(days=off)
                    if d < start_date:
                        continue
                    if until_epoch is not None and ts(d) > until_epoch:
                        excluded = exdates or set()
                        return [ts(day) for day in days if ts(day) not in excluded]
                    days.append(d)
                    if len(days) >= limit:
                        break
                week += 1
        else:
            for i in range(limit):
                d = start_date + timedelta(weeks=i * interval)
                if until_epoch is not None and ts(d) > until_epoch:
                    break
                days.append(d)
    elif freq == "MONTHLY":
        i = 0
        while len(days) < limit:
            month_index = start_date.year * 12 + (start_date.month - 1) + i * interval
            year, month = divmod(month_index, 12)
            month += 1
            d = date(year, month, min(start_date.day, monthrange(year, month)[1]))
            if until_epoch is not None and ts(d) > until_epoch:
                break
            days.append(d)
            i += 1
    else:  # YEARLY
        for i in range(limit):
            year = start_date.year + i * interval
            day = start_date.day
            if start_date.month == 2 and day == 29:
                day = min(day, monthrange(year, 2)[1])
            d = date(year, start_date.month, day)
            if until_epoch is not None and ts(d) > until_epoch:
                break
            days.append(d)

    excluded = exdates or set()
    return [ts(d) for d in days if ts(d) not in excluded]


def _event_occurrences(
    props: dict[str, str],
    tzids: dict[str, str | None],
    all_day: bool,
) -> list[tuple[float, float]]:
    start = (
        _parse_ics_date(props["DTSTART"])
        if all_day
        else _parse_ics_datetime(props["DTSTART"], tzids.get("DTSTART"))
    )
    end = _absence_end(props, start, all_day, tzids)
    rrule = props.get("RRULE")
    if not rrule:
        return [(start, end)]
    rule = _parse_rrule(rrule)
    until = None
    if "UNTIL" in rule:
        until_value = rule["UNTIL"].strip()
        if len(until_value) == 8 and until_value.isdigit():
            until = _parse_ics_date(until_value) + 86400.0 - 1.0
        else:
            until = _parse_ics_datetime(until_value, tzids.get("DTSTART"))
    exdates = {
        _parse_ics_date_or_datetime(v, tzids.get("EXDATE"))
        for v in props.get("EXDATE", "").split(",")
        if v
    }
    duration = max(end - start, 0.0)
    return [(s, s + duration) for s in _expand_rrule(start, rule, until, exdates)]


def _classify_vevents(
    vevents: list[tuple[dict[str, str], dict[str, str | None], bool]],
) -> tuple[list[Event], list[Absence]]:
    events: list[Event] = []
    absences: list[Absence] = []
    for props, tzids, all_day in vevents:
        try:
            if _is_absence(props, all_day):
                absences.extend(
                    Absence(start=start, end=end, kind=_absence_kind(props))
                    for start, end in _event_occurrences(props, tzids, all_day)
                )
            elif not all_day:
                events.extend(
                    Event(
                        timestamp=start,
                        kind="meeting",
                        duration_minutes=max((end - start) / 60.0, 1.0),
                        meta={"title": props.get("SUMMARY", "")},
                    )
                    for start, end in _event_occurrences(props, tzids, all_day)
                )
            else:
                # Non-absence all-day event (e.g. conference, workshop)
                events.extend(
                    Event(
                        timestamp=start,
                        kind="meeting",
                        duration_minutes=480.0,
                        meta={"title": props.get("SUMMARY", ""), "all_day": True},
                    )
                    for start, end in _event_occurrences(props, tzids, all_day)
                )
        except (ValueError, TypeError) as exc:
            logger.warning("skipping malformed calendar event: %s", exc)
    return events, absences


def parse_calendar(path: Path) -> tuple[list[Event], list[Absence]]:
    return _classify_vevents(_vevents(path))


def parse_calendar_text(text: str) -> tuple[list[Event], list[Absence]]:
    return _classify_vevents(_vevents_text(text))


def parse_ics(path: Path) -> list[Event]:
    events, _ = parse_calendar(path)
    return events


def parse_absences(path: Path) -> list[Absence]:
    _, absences = parse_calendar(path)
    return absences
