# G1 — independent official-page deadline verification

Access: 2026-11-09 09:27–09:28 UTC / 11:27–11:28 Europe/Kyiv, with session clock recorded at 09:27:58 UTC immediately after the final Overview observation. Public official page; the available browser showed Login and Sign Up. Authenticated access was not needed to read the absolute Close tooltip. No authenticated Submit-page inspection is claimed.

Source: [official competition Overview](https://www.kaggle.com/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026/overview), Overview tab, “Close” field. The rendered text was **“2 days to go”**. Its native `title` tooltip was exactly:

> Tue Nov 10 2026 23:55:00 GMT+0200 (Eastern European Standard Time)

The competition header showed the same tooltip. This is a browser-rendered timezone representation: explicit GMT+0200, Eastern European Standard Time; it does not establish that Kaggle stores deadlines in that timezone.

- Normalized UTC: **2026-11-10T21:55:00Z**.
- Europe/Kyiv: **2026-11-10T23:55:00+02:00** (EET), checked using Python zoneinfo.
- [Official Rules tab](https://www.kaggle.com/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026/rules): inspected the competition-specific rules; no separate absolute final-selection/LMS deadline was exposed. Header repeats the same Close tooltip. No later selection window is inferred.

Result: official Close independently verified, agreeing with the earlier G1 evidence. User explicitly accepted public-page verification in this session. Authenticated verification was not performed and is not required for this exposed timestamp. No gate or board row was edited. Claude/Nazar can incorporate this source record later. The historical 23:59 student recollection is not used.

Evidence method: read the rendered DOM’s native title attributes for visible relative-date elements on Overview, then inspect Rules. No hidden application state, third-party deadline, account identifiers or credentials were collected. No screenshot was saved; text evidence avoids including account information. The initial Overview observation was made before the clock record 09:23:51 UTC; the final Overview check immediately preceded the 09:27:58 UTC record. Access is recorded as a minute interval rather than an invented exact subsecond observation.
